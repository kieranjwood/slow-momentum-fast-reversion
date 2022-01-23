import multiprocessing
import os

from settings.default import QUANDL_TICKERS, CPD_QUANDL_OUTPUT_FOLDER_DEFAULT

N_WORKERS = len(QUANDL_TICKERS)

if not os.path.exists(CPD_QUANDL_OUTPUT_FOLDER_DEFAULT):
    os.mkdir(CPD_QUANDL_OUTPUT_FOLDER_DEFAULT)

all_processes = [
    f'python script_cpd_example.py "{ticker}" "{os.path.join(CPD_QUANDL_OUTPUT_FOLDER_DEFAULT, ticker + ".csv")}" "1990-01-01" "2019-12-31"'
    for ticker in QUANDL_TICKERS
]
process_pool = multiprocessing.Pool(processes=N_WORKERS)
process_pool.map(os.system, all_processes)
