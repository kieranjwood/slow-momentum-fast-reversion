import os

CPD_LBWS = [10, 21, 63, 126, 256]
CPD_DEFAULT_LBW = 21
USE_KM_HYP_TO_INITIALISE_KC = True
CPD_OUTPUT_FOLDER_DEFAULT = os.path.join(
    "data", f"commodities_cpd_{CPD_DEFAULT_LBW}lbw"
)

FEATURES_FILE_PATH_DEFAULT = os.path.join(
    "data", f"features_cpd_{CPD_DEFAULT_LBW}lbw.csv"
)

# commodities from Yahoo Finance
COMMODITIES_TICKERS = [
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