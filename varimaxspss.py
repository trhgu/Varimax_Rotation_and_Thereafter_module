
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from factor_analyzer import FactorAnalyzer, Rotator


class PcaVarimax:
    """
    1. 데이터 프레임 타입의 변수 'x'를 받아 들입니다.
    2. pca() 함수는 고유값을 반환합니다. 
    3. loading(n) 값을 도출하기 위해서 원하는 차수(n)를 입력하셔야 합니다.
    4. varimax() 함수는 varimax rotate가 적용된 결과를 반환 합니다.
    """

    def __init__(self, x):
        self.x = x

    def pca(self):
        # 문항간 상관계수 도출
        self.numCorr = self.x.corr()
        # 고유치와 고유벡터 도출
        eVal_corr, eVec_corr = np.linalg.eig(self.numCorr)
        # 고유치의 크기가 큰 순으로 고유치 및 고유벡터 정리(argsort()[::-1]제일 큰 값부터 순서대로 데이터의 인덱스값을 출력해줌)
        idx_corr = eVal_corr.argsort()[::-1]
        self.eVal_corr = eVal_corr[idx_corr]
        eVec_corr = eVec_corr[:, idx_corr]
        # 고유치와 고유벡터 도출 # 고유치의 크기가 큰 순으로 고유치 및 고유벡터 정리 한번에
        U, S, VT = np.linalg.svd(self.numCorr)
        # 고유백터 합이 1이하인 수에 -1을 곱해준다

        def flip_vector_sign(eVec):
            # shape는 어레이의 행과 열 개수를 알려준다 .shape[0]는 열 개수
            for i in range(eVec.shape[1]):
                if(eVec[:, i].sum() < 0):
                    eVec[:, i] = -1*eVec[:, i]
            return eVec

        self.V = flip_vector_sign(VT.T)
        # 총 분산 설명
        exp = eVal_corr*100/np.sum(eVal_corr)
        accSum = np.cumsum(exp)  # 누적합
        xcolumns = self.x.columns
        pcNum = list(range(1, len(xcolumns)+1))  # 상수항은 항시 제거되야한다.
        data = np.array([pcNum, eVal_corr, exp, accSum])
        eigenValues = pd.DataFrame(
            data.T, columns=['PC#', 'Eigenvalue', '% of Varian Exp', 'Cumulative %'])
        eigenNumbers = eigenValues.copy()
        format_mapping = {'PC#': '{:,.0f}', 'Eigenvalue': '{:,.3f}',
                          '% of Varian Exp': '{:.3f}', 'Cumulative %': '{:.3f}%'}

        for key, value in format_mapping.items():
            eigenValues[key] = eigenValues[key].apply(value.format)

        eachExp = eigenNumbers.iloc[:, 2]

        plt.figure(figsize=(6, 6))
        plt.bar(pcNum, accSum, width=0.5, color='cyan',
                alpha=0.2, label="Cumulative %")
        plt.plot(pcNum, eachExp, label="% of Variance Explained")
        plt.plot(pcNum, eachExp, 'ro', label='_nolegend_')
        plt.xlabel("Principal Components")
        plt.ylabel("% of Variance Explained")
        plt.title("% of Varianve Explained by PCs", fontsize=16)
        plt.legend(loc='upper left')

        print(eigenValues)

    def loadings(self, n):
        eVec_corr3 = self.V[:, :n]
        eVal_corr3 = self.eVal_corr[:n]
        self.loading3 = eVec_corr3*np.sqrt(eVal_corr3)
        loading3T = self.loading3.round(n).T
        index = []
        for index_n in range(1, n+1):
            a = "PC" + str(index_n)
            index.append(a)
        loadingDF = pd.DataFrame(loading3T, index=[index], columns=[
            self.x.columns])
        pcScoreCoef = np.linalg.inv(self.numCorr)@(self.loading3)
        zScore = (self.x - self.x.mean())/self.x.std()
        pcScore = zScore.dot(pcScoreCoef)
        print(loadingDF.T)
        return pcScore.round(5)

    def varimax(self):
        fa = FactorAnalyzer(rotation=None)
        rotator = Rotator()
        a = rotator.fit(self.loading3)
        return a.rotation_, a.loadings_
