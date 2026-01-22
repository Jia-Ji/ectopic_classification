from pyPPG.example import ppg_example
import numpy as np
from pyPPG import PPG, Fiducials, Biomarkers
from pyPPG.datahandling import load_data, plot_fiducials, save_data
import pyPPG.preproc as PP
import pyPPG.fiducials as FP
import pyPPG.biomarkers as BM
import pyPPG.ppg_sqi as SQI
from scipy.signal import resample
import os
import pandas as pd
from tqdm import tqdm
import warnings
try:
    from .biomarker_utils import convert_npy_to_mat, merge_ppg_segment_csvs, delete_empty_dirs
except ImportError:
    # Fallback for when running as a script directly
    from biomarker_utils import convert_npy_to_mat, merge_ppg_segment_csvs, delete_empty_dirs
try:
    from pandas.errors import SettingWithCopyWarning
except Exception:
    class SettingWithCopyWarning(Warning):
        pass

class BiomarkerExtractor():
    def __init__(self, fs, mat_save_path, temp_mat_save_path, savingfolder, start_sig=0, end_sig=-1, pad_width= 0, tile_reps=1, use_tk=False):
        self.fs = fs
        self.savingfolder = savingfolder
        if not os.path.exists(self.savingfolder):
            os.mkdir(self.savingfolder)
        # Ensure paths are joined properly
        self.mat_save_path = os.path.join(savingfolder, mat_save_path)
        self.temp_mat_save_path = os.path.join(savingfolder, temp_mat_save_path)
        os.makedirs(self.mat_save_path, exist_ok=True)
        os.makedirs(self.temp_mat_save_path, exist_ok=True)
        self.start_sig = start_sig
        self.end_sig = end_sig
        self.pad_width = pad_width
        self.tile_reps = tile_reps
        self.use_tk = use_tk
        self.index = 0

    def create_ppg(self, s_path, start_sig=0, end_sig=-1, use_tk=False):
        signal0 = load_data(data_path=s_path, start_sig=start_sig, end_sig=end_sig, use_tk=use_tk)
        signal0.v = signal0.v
        # Ensure sampling frequency is a scalar float (some .mat files store Fs as arrays)
        try:
            signal0.fs = float(np.squeeze(signal0.fs))
        except Exception:
            signal0.fs = float(signal0.fs)

        # PPG signal processing
        signal0.filtering = True # whether or not to filter the PPG signal
        signal0.fL=0.5000001 # Lower cutoff frequency (Hz)
        signal0.fH=12 # Upper cutoff frequency (Hz)
        signal0.order=4 # Filter order
        signal0.sm_wins={'ppg':50,'vpg':10,'apg':10,'jpg':10} # smoothing windows in millisecond for the PPG, PPG', PPG", and PPG'"

        prep = PP.Preprocess(fL=signal0.fL, fH=signal0.fH, order=signal0.order, sm_wins=signal0.sm_wins)
        signal0.ppg, signal0.vpg, signal0.apg, signal0.jpg = prep.get_signals(s=signal0)

        # Initialise the correction for fiducial points
        corr_on = ['on', 'dn', 'dp', 'v', 'w', 'f']
        correction=pd.DataFrame()
        correction.loc[0, corr_on] = True
        signal0.correction=correction

        # Create a PPG class
        s = PPG(s=signal0, check_ppg_len=False)

        return s
    
    def filter_temp_fiducials(self, df: pd.DataFrame, s_length: int, pad_width: int) -> pd.DataFrame:
        df_copy = df.copy()

        for col in df_copy.columns:
            def adjust(val):
                    if pd.isna(val):
                        return pd.NA
                    new_val = val - pad_width
                    return new_val if 0 <= new_val <= s_length-1 else pd.NA
            df_copy[col] = df_copy[col].apply(adjust)

        fiducial_cols = df_copy.columns
        df_filtered = df_copy.dropna(subset=fiducial_cols, how="all").reset_index(drop=True)
        df_filtered.index.name = "Index of pulse"

        for column in df_filtered.columns:
            df_filtered[column] = pd.to_numeric(df_filtered[column], errors='coerce').astype("Int64")

        return df_filtered
    
    def extract_biomarkers_from_fiducial_points(self, ppg_signal: np.ndarray, signal_index: int):

        signal_arr = np.asarray(ppg_signal).squeeze()
        signal_arr = np.nan_to_num(signal_arr, nan=0.0, posinf=0.0, neginf=0.0).astype(float)

        # Write out non-padded and padded (temp) .mat files for processing
        signal = convert_npy_to_mat(signal_arr, fs=self.fs, pad=False, pad_width=self.pad_width, tile=False, tile_reps=self.tile_reps, save_path=self.mat_save_path, signal_index=signal_index)
        temp_signal = convert_npy_to_mat(signal_arr, fs=self.fs, pad=True, pad_width=self.pad_width, tile=True, tile_reps=self.tile_reps, save_path=self.temp_mat_save_path, signal_index=signal_index)

        signal_path = os.path.join(self.mat_save_path, f"segment_{signal_index}.mat")
        temp_signal_path = os.path.join(self.temp_mat_save_path, f"temp_segment_{signal_index}.mat")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            warnings.simplefilter("ignore", category=SettingWithCopyWarning)
            s = self.create_ppg(s_path=signal_path, start_sig=self.start_sig, end_sig=self.end_sig, use_tk=self.use_tk)
            temp_s = self.create_ppg(s_path=temp_signal_path, start_sig=self.start_sig, end_sig=self.end_sig, use_tk=self.use_tk)

            fpex = FP.FpCollection(temp_s)
            temp_fiducials = fpex.get_fiducials(temp_s)

            fiducials = self.filter_temp_fiducials(temp_fiducials, len(s.ppg), self.pad_width)

            # Create a fiducials class
            fp = Fiducials(fiducials)
            fp_new = fp.get_fp() + s.start_sig
            fp_new.index = fp_new.index + 1

            # fp_new = Fiducials(fp.get_fp() + s.start_sig).get_fp()
            # fp_new.index = fp_new.index + 1

            # derive SQI
            annotations, beatSQI, ppgSQI = self.calculate_SQI(fp, s)
            sqi = float(ppgSQI)

            # transform beat-level SQI to sample-level SQI
            sqi_sample = self.transform_beat_sqi_to_sample(beatSQI, annotations , s.ppg)

            hr = self.estimate_HR(fp, s)

            bm_vals, bm_stats = self.derive_biomarkers(fp, s)
            hrv = self.estimate_HRV(bm_vals)

            # update index tracker
            self.index = max(self.index, signal_index + 1)

            return sqi, sqi_sample, hr, fp_new, bm_stats, hrv
    
    def estimate_HRV(self, bm_vals):
        # Extract Tpp (peak-to-peak interval in seconds)
        Tpp = bm_vals["ppg_sig"]["Tpp"].values.astype(float)    # seconds
        # Convert to NN intervals in milliseconds
        NN = Tpp * 1000.0

        diffs = np.diff(NN)
        # RMSSD
        RMSSD = np.sqrt(np.mean(diffs**2))
         # SDSD (std of successive differences)
        SDSD = np.std(diffs, ddof=1)
        print('Estimated HRV - RMSSD: ', RMSSD, ' ms, SDSD: ', SDSD, ' ms')

        return {"RMSSD": RMSSD, "SDSD": SDSD}
        
    def estimate_HR(self, fp, s):
        num_beats=len(fp.sp)  # number of the beats
        duration_seconds=len(s.ppg)/s.fs  # duration in seconds
        HR = (num_beats / duration_seconds) * 60 # heart rate
        print('Estimated HR: ',HR,' bpm' )
        return HR
    
    def calculate_SQI(self, fp: Fiducials, s:PPG):
        annotations = fp.sp.copy()
        # Convert to numpy if it’s pandas
        if isinstance(annotations, pd.Series):
            annotations = annotations.dropna().values

        # Remove invalid annotations (e.g., less than 1 or larger than PPG length)
        annotations = annotations[(annotations > 0) & (annotations < len(s.ppg))]

        beatSQI = SQI.get_ppgSQI(s.ppg, s.fs, annotations)
        ppgSQI = round(np.nanmedian(SQI.get_ppgSQI(s.ppg, s.fs, annotations)) * 100, 2)
        # ppgSQI = round(np.nanmean(SQI.get_ppgSQI(s.ppg, s.fs, annotations)) * 100, 2)
        print('Mean PPG SQI: ', ppgSQI, '%')
        return annotations, beatSQI, ppgSQI
    
    def derive_biomarkers(self, fp, s):
        bmex = BM.BmCollection(s, fp)
        
        bm_defs, bm_vals, bm_stats = bmex.get_biomarkers()
        # bm_defs, bm_vals = bmex.get_biomarkers()
        # tmp_keys=bm_stats.keys()
        # print('Statistics of the biomarkers:')
        # for i in tmp_keys: 
        #     print(i,'\n',bm_stats[i])
        
        return bm_vals, bm_stats

    def transform_beat_sqi_to_sample(self, beatsqi, annotation, ppg):
        """
        Transform beat-level SQI to sample-level SQI.

        :param beatsqi: Beat-level SQI array
        :param annotation: Annotation array (peak locations)
        :param ppg: PPG signal array
        :param combined_df: DataFrame to save the sample-level SQI
        """
        # Step 1: Clip range to remove NaNs and define beatSQI in the range of 0 and 1
        beatsqi = np.nan_to_num(beatsqi, nan=0.0)
        beatsqi = np.clip(beatsqi, 0.2, 1.0)

        # Step 2: Expand beat SQI to sample-level SQI
        sqi_sample = np.zeros(len(ppg))
        for i in range(len(annotation) - 1):
            start = int(annotation[i])
            end = int(annotation[i + 1])
            sqi_sample[start:end] = beatsqi[i]

        # Handle edges
        sqi_sample[:int(annotation[0])] = 0.2
        sqi_sample[int(annotation[-1]):] = 0.2

        return sqi_sample

    



    
