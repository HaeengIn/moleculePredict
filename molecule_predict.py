# 모듈 임포트
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from rdkit import Chem
from rdkit.Chem import Descriptors

# SMILES 리스트 정의
smilesList = ["O", "O=C=O", "C", "CCO", "CC(=O)O", "C(C1C(C(C(C(O1)O)O)O)O)O", "CC(=O)C", "CN1C=NC2=C1C(=O)N(C(=O)N2C)", "C(C(CO)O)O"]
X = []

# 분자량 계산
for s in smilesList:
    mol = Chem.MolFromSmiles(s)
    X.append([Descriptors.MolWt(mol)])

# 예시 녹는점 (물, 이산화탄소, 메탄, 에탄올, 아세트산)
y = [0, 0, 0, 78, 118, 146, -95, 238, 18]

# 데이터 분할 (80% 학습, 20% 테스트)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 모델 학습
model = RandomForestRegressor()
model.fit(X_train, y_train)

# 예측 및 평가
y_pred = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))