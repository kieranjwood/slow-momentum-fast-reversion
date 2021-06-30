import multiprocessing
import os

OUTPUT_FOLDER = os.path.join("data", "commodities")

# commodities from Yahoo Finance
TICKERS = [
    "CC=F",
    "CL=F",
    "CT=F",
    "ES=F",
    "GC=F",
    "GF=F",
    "HE=F",
    "HG=F",
    "HO=F",
    "KC=F",
    "KE=F",
    "LBS=F",
    "LE=F",
    "MGC=F",
    "NG=F",
    "NQ=F",
    "OJ=F",
    "PA=F",
    "PL=F",
    "RB=F",
    "RTY=F",
    "SB=F",
    "SI=F",
    "SIL=F",
    "YM=F",
    "ZB=F",
    "ZC=F",
    "ZF=F",
    "ZL=F",
    "ZM=F",
    "ZN=F",
    "ZO=F",
    "ZR=F",
    "ZS=F",
    "ZT=F",
]

N_WORKERS = len(TICKERS)

all_processes = [
    f'python script_cpd_example.py "{ticker}" "{os.path.join(OUTPUT_FOLDER, ticker + ".csv")}" "2000-01-01" "2019-12-31"'
    for ticker in TICKERS
]
process_pool = multiprocessing.Pool(processes=N_WORKERS)
process_pool.map(os.system, all_processes)
