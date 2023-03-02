from datetime import datetime as dt
import pandas as pd

SUMMARY_CSV_COLS = ["module", "hashed_uid", "success", "log", "start_time", "end_time"]


class PatientLogger:
    """Class to document a patient's pipeline progress in a personalised CSV file"""

    def __init__(self, pat_id, data_directory, force=True):
        self.pat_id = pat_id
        self.data_directory = data_directory
        self.start_time = dt.now()

        # create pat dir if not yet created
        pat_directory = self.data_directory.joinpath(pat_id)
        pat_directory.mkdir(exist_ok=True)

        self.summary_csv_path = pat_directory.joinpath("summary.csv")

        # create patient csv if not already created
        if not self.summary_csv_path.exists() or force:
            df_pat_log = pd.DataFrame(columns=SUMMARY_CSV_COLS)
            df_pat_log.to_csv(self.summary_csv_path, index=False)

    def log_module_error(self, module, hashed_uid, error_log):
        """Function to log errors for a specific pydicer module

        Args:
            module (str): pydicer module to log error for in CSV
            error (str): error to log in CSV
        """
        end_time = dt.now()
        df_error = pd.DataFrame(
            [[module, hashed_uid, 1, error_log, self.start_time, end_time]],
            columns=SUMMARY_CSV_COLS,
        )
        df_error.to_csv(self.summary_csv_path, header=False, mode="a", index=False)

    def eval_module_process(self, module, hashed_uid):
        """Function to log if any patient had issues for a specific pydicer module

        Args:
            module (str): pydicer module to check if no errors were generated for all patients
        """

        end_time = dt.now()
        df_summary = pd.read_csv(self.summary_csv_path)
        df_summary_mod = df_summary[
            (df_summary.module == module)
            & (df_summary.success == 1)
            & (df_summary.hashed_uid == hashed_uid)
        ]
        if len(df_summary_mod) == 0:
            df_final_summary = pd.DataFrame(
                [[module, hashed_uid, 0, "", self.start_time, end_time]], columns=SUMMARY_CSV_COLS
            )
            df_final_summary.to_csv(
                self.summary_csv_path,
                header=False,
                mode="a",
                index=False,
            )
