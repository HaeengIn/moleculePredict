import os, sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor

INPUT_XLSX = "BradleyMeltingPointDataset.xlsx"
FP_BITS = 1024
TEST_SIZE = 0.20
VALID_SIZE = 0.10
RANDOM_STATE = 42
SAMPLE_FRAC = 0.6

if not os.path.exists(INPUT_XLSX):
    print("파일을 찾을 수 없습니다:", INPUT_XLSX); sys.exit(1)

df = pd.read_excel(INPUT_XLSX)
df.columns = df.columns.str.strip()
if 'smiles' not in df.columns or 'mpC' not in df.columns:
    print("smiles 또는 mpC 컬럼이 없습니다."); sys.exit(1)

df = df[df['mpC'].notna()].copy()
if 'donotuse' in df.columns:
    df = df[df['donotuse'].isna() | (df['donotuse'] == '')].copy()
df = df.reset_index(drop=True)

valid_idx = []
for i, smi in tqdm(df['smiles'].items(), total=len(df), desc="SMILES 정제"):
    if not isinstance(smi, str) or smi.strip() == "":
        continue
    mol = Chem.MolFromSmiles(smi, sanitize=False)
    if mol is None:
        continue
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        continue
    valid_idx.append(i)

df = df.loc[valid_idx].reset_index(drop=True)
if len(df) < 10:
    print("유효한 분자가 너무 적습니다."); sys.exit(1)

def calc_descriptors(mol):
    return np.array([
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        rdMolDescriptors.CalcNumRings(mol),
        rdMolDescriptors.CalcFractionCSP3(mol)
    ], dtype=float)

def mol_to_fp_array(mol, nBits=FP_BITS, radius=2):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    onbits = list(fp.GetOnBits())
    arr = np.zeros((nBits,), dtype=np.uint8)
    if len(onbits) > 0:
        arr[onbits] = 1
    return arr

desc_list, fp_list = [], []
for smi in tqdm(df['smiles'], desc="특성 생성"):
    mol = Chem.MolFromSmiles(smi)
    desc_list.append(calc_descriptors(mol))
    fp_list.append(mol_to_fp_array(mol))

X_desc = np.vstack(desc_list)
X_fp = np.vstack(fp_list)
y = df['mpC'].astype(float).values

if 0 < SAMPLE_FRAC < 1.0:
    n_total = X_desc.shape[0]
    n_sel = max(200, int(n_total * SAMPLE_FRAC))
    idx_sel = np.random.RandomState(RANDOM_STATE).choice(n_total, n_sel, replace=False)
    X_desc = X_desc[idx_sel]
    X_fp = X_fp[idx_sel]
    y = y[idx_sel]

X = np.hstack([X_desc, X_fp])

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=VALID_SIZE, random_state=RANDOM_STATE)

model = LGBMRegressor(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=64,
    n_jobs=-1,
    random_state=RANDOM_STATE
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"테스트 RMSE: {rmse:.4f} °C")

def prepare_single(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError("SMILES 파싱 실패: " + smi)
    d = calc_descriptors(mol).reshape(1, -1)
    fp = mol_to_fp_array(mol).reshape(1, -1)
    return np.hstack([d, fp])

aspirin_smiles = "CC(=O)Oc1ccccc1C(=O)O"
caffeine_smiles = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"

for name, smi in [("Aspirin", aspirin_smiles), ("Caffeine", caffeine_smiles)]:
    try:
        Xnew = prepare_single(smi)
        pred = model.predict(Xnew)[0]
        print(f"{name} ({smi}) 예상 녹는점: {pred:.2f} °C")
    except Exception as e:
        print(f"{name} 예측 오류:", e)
