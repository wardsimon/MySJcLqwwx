import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import gridspec

trading_days = 252 # Number of trading days a year


class Predictors:
    """
    Predictors encapsulates all technical indicators which can be used for modeling.
    """
    def __init__(self, data):
        """
        :param data: pandas DataFrame with stock information from Yahoo finance
        
        :return: nothing 

        Construct a new Predictors object
        """
        self.data = data
        self._days = 5
        self._min_shift = 0.03
        self._shortPredictors = self.make_short()
        self._long_predictors = self.make_long()

    def make_short(self):
        """
        Makes all technical indicators which are not based on user input.
        """
        # Note, we have not taken into consideration inflation :-/
        dat = self.data
        df = pd.DataFrame({
            # Short rolling mean
            "r.lag1": self.make_splits(30, False),
            # Long rolling mean
            "r.lag2": self.make_splits(60, False),
            # Price / Volume
            "vol/px": (dat["Volume"] / dat["Close"]) / 1E6,
            # The monthly volatility
            "mon_vol": np.round(dat["Close"].rolling(window=20, center=False).std(), 2) * np.sqrt(trading_days),
            # 12 month earnings (as %)
            "12mE": (dat["Close"] - dat["Close"].shift(trading_days)) / dat["Close"].shift(trading_days),
            # 10 day volume change
            "mom10": (dat["Volume"] - dat["Volume"].shift(10)) / dat["Volume"].shift(10),
            # 10 day rolling volatility
            "vol_mon": np.round(dat["Volume"].rolling(window=10, center=False).std(), 2) * np.sqrt(trading_days),
            # Price
            "price": dat['Close'],
            # Standard MACD (see documentation)
            "MACD": self.MACD(),
            # MACD histogram (see documentation)
            "MACD_l": self.MACD_signal_line(hist=True),
            # Stochastic Oscillator (see documentation)
            "Stok0": self.make_stokO(),
            # Reverse Stochastic Oscillator (see documentation)
            "RStok0": self.make_RstokO(),
            # True range
            "TR": self.calc_tr(),
            # The number of consecutive days where the price has risen
            "DP": self.days_C_prev()
        })
        return df

    def make_long(self):
        """
        Makes all technical indicators which are based on time. 
        Time is in days and is set by the user
        """
        if self.days is None:
            print("Supply a time period")
            return
        # Get the data and days
        ndays = self.days
        dat = self.data

        # Rolling volatility
        temp = np.round(dat["Close"].rolling(window=ndays, center=False).std(), 2) * np.sqrt(trading_days)

        long_predictors = pd.DataFrame({
            "meanfractal": pd.DataFrame(self._rep_predictors(ndays)).sum(1, skipna=False) / ndays,
            # Local Volititlity
            "loc_vol": temp,
            # Change in volume
            "dvol": (temp - temp.shift(1)) / temp.shift(1),  # Change in local volatility
            # Rolling momentum
            "mom": (dat['Close'] - dat["Close"].shift(ndays)).map(self._logic),
            # The rolling true range
            "ATR": self.calc_atr(ndays),
            # Rolling number of days above a baseline
            "PHH": self.make_percentage_higher(ndays,self._min_shift)
        })
        return long_predictors

    def _rep_predictors(self, itter):
        dat = self.data
        df = pd.DataFrame()
        for count in range(1, itter + 1):
            dat['Direction'] = np.where(dat['Close'].diff(count) > 0, 1, 0)
            dat['Abs'] = dat['Close'].diff(count).abs()
            dat['Volatility'] = dat.Close.diff().abs().rolling(count).sum()
            dat['Fractal'] = dat['Abs'] / dat['Volatility'] * dat['Direction']
            df = pd.concat([df, dat['Fractal']], axis=1)
        return df

    def _rep_predictorsE(self, itter):
        dat = self.data
        df = pd.DataFrame()

        for count in range(1, itter + 1):
            dat['Direction'] = np.where(dat['Close'].diff(count) > 0, 1, 0)
            dat['Abs'] = dat['Close'].diff(count).abs()
            dat['Volatility'] = dat.Close.diff().abs().rolling(count).sum()
            dat['Fractal'] = dat['Abs'] / dat['Volatility'] * dat['Direction']
            df = pd.concat([df, dat['Fractal']], axis=1)
        return df

    def _logic(self, x):
        if x < 0:
            x = -1
        elif x > 0:
            x = 1
        return x

    def make_splits(self, days=20, inplace=True):
        dat = self.data
        if inplace:
            dat['%id' % days] = np.round(dat["Close"].rolling(window=days, center=False).mean(), 2)
        else:
            return np.round(dat["Close"].rolling(window=days, center=False).mean(), 2)

    def make_esplits(self, days=20, inplace=True,C = None):
        if C is None:
            C = self.data
        if inplace:
            self.data['%id' % days] = C.Close.ewm(span=days).mean()
        else:
            return C.Close.ewm(span=days).mean()

    def e_filter(self,days):
        p = self.data.columns.values
        for val in p:
            self.data[val] = self.data[val].ewm(span=days).mean()

    def make_bollinger(self, days=20, fac=2, inplace=True):
        dat = self.data
        temp = dat["Close"].rolling(window=days, center=False).std()
        if inplace:
            dat['%iBol_U' % days] = dat["Close"].rolling(window=days, center=False).mean() + fac * temp
            dat['%iBol_L' % days] = dat["Close"].rolling(window=days, center=False).mean() - fac * temp
        else:
            return (dat["Close"].rolling(window=days, center=False).mean() + fac * temp,
                    dat["Close"].rolling(window=days, center=False).mean() - fac * temp)

    def make_sharpR(self):
        dat = self.data

    def make_stokO(self, period = 14, C = None):
        L14 = self.data.Close.rolling(period).min()
        H14 = self.data.Close.rolling(period).max()
        if C is None:
            C = self.data
        return 100*(C.Close - L14) / (H14 - L14)

    def make_RstokO(self, period = 14, C = None):
        L14 = self.data.Close.rolling(period).min()
        H14 = self.data.Close.rolling(period).max()
        if C is None:
            C = self.data
        return -100*(H14 - C.Close)/(H14 - L14)

    def stokO_signal(self,upper = 80, lower = 20, days = 3, period = 14, K = None, C = None):
        if K is None:
            if C is None:
                K = self.make_stokO(period=period)
            else:
                K = self.make_stokO(period=period,C=C)
        D = K.rolling(days).mean()
        def fun_D(x):
            if x > upper:
                x = 1
            elif x < lower:
                x = -1
            else:
                x = 0
            return x
        return D.map(fun_D)

    def movingA_signal(self,fast=20,slow=150,C = None):
        if C is None:
            C = self.data
        R1 = C.Close.rolling(fast).mean()
        R2 = C.Close.rolling(slow).mean()
        A = R1 - R2
        def fun_A(x):
            if x > 0:
                x = 1
            elif x < 0:
                x = -1
            else:
                x = 0
            return x
        return A.map(fun_A)

    def MACD(self,fast = 12, slow = 26,C = None):
        if C is None:
            C = self.data
        R1 = self.make_esplits(fast,inplace=False,C=C)
        R2 = self.make_esplits(slow, inplace=False, C=C)
        return R1 - R2

    def MACD_signal_line(self,period = 9, hist = False, signal = False, MACD = None, fast = 12, slow = 26, C = None):
        if MACD is None:
            MACD = self.MACD(fast,slow,C)
        SL = MACD.ewm(span=period).mean()
        if hist:
            if signal:
                temp = (MACD - SL)
                temp[temp > 0] = 1
                temp[temp < 0] = -1
                return temp
            else:
                return MACD - SL
        else:
            return SL


    def movAVG_BuySell(self,periodM = 9, fastM = 12, slowM = 26, fastA = 20, slowA = 150, C = None,
                    upperS = 80, lowerS = 20, daysS = 3, periodS = 14, K = None):
        if C is None:
            C = self.data
        C["Avg_Sig"] = self.movingA_signal(fast=fastA,slow=slowA,C = C)
        C["stok_Sig"] = self.stokO_signal(upper=upperS,lower=lowerS,days=daysS,period=periodS,C = C, K=K)
        C["MACD_Sig"] = self.MACD_signal_line(period=periodM,hist=True,signal=True,fast=fastM,slow=slowM)

        Buy  = (C.Avg_Sig == 1) & (C.stok_Sig ==1) & (C.MACD_Sig==1)
        Sell = (C.Avg_Sig == -1) & (C.stok_Sig ==-1) & (C.MACD_Sig==-1)

        C["regime"] = (C.Avg_Sig + C.stok_Sig + C.MACD_Sig)
        C["regime"].ix[C["regime"] == 2] = 0
        C["regime"].ix[C["regime"] == 1] = 0
        C["regime"].ix[C["regime"] == 0] = 0
        C["regime"].ix[C["regime"] == -1] = 0
        C["regime"].ix[C["regime"] == -2] = 0
        C["regime"].ix[C["regime"] == 3] = 1
        C["regime"].ix[C["regime"] == -3] = -1


        CC = C[C.regime != 0].copy()
        temp = CC['regime'].ix[-1]
        CC['regime'].ix[-1] = 0
        CC['signal'] = np.sign(CC['regime'] - CC['regime'].shift(1))
        CC['regime'].ix[-1] = temp

        gs = gridspec.GridSpec(4, 1, height_ratios=[3, 1, 1, 1])
        ax1 = plt.subplot(gs[0])
        C.Close.plot(label='Price')
        C.Close.rolling(fastA).mean().plot(label='F_Mean')
        C.Close.rolling(slowA).mean().plot(label='S_Mean')
        ax2 = plt.subplot(gs[1], sharex=ax1)
        ax2.plot(np.NAN,np.NAN)
        K = self.make_stokO(period=periodS)
        D = K.rolling(daysS).mean()
        K.plot(label='Fast_Osc')
        D.plot(label='Fast_Osc3')
        ax2.xaxis.set_ticklabels([])
        ax3 = plt.subplot(gs[2], sharex=ax1)
        MA = self.MACD_signal_line(period=periodM,hist=True,signal=False,fast=fastM,slow=slowM)
        plt.bar(MA.index,MA.values, align='center',label="MACD")
        ax4 = plt.subplot(gs[3], sharex=ax1)
        CC['signal'].plot()

        moves = pd.concat([
            pd.DataFrame({"Price": CC.loc[CC["signal"] == 1, "Close"],
                          "Regime": CC.loc[CC["signal"] == 1, 'regime'],
                          "Signal": "Buy"}),
            pd.DataFrame({"Price": CC.loc[CC["signal"] == -1, "Close"],
                          "Regime": CC.loc[CC["signal"] == -1, 'regime'],
                          "Signal": "Sell"})
        ])
        moves.sort_index(inplace=True)
        if moves.ix[0].Regime == -1:
            moves = moves.ix[1:]
        return moves


    def days_C_prev(self):
        sw = 1*(self.data.Close.shift(1).div(self.data.Close)> 1)
        for i in range(0,len(sw)):
            if bool(sw.iloc[i]):
                sw.iloc[i] = sw.iloc[i-1]+1
        return sw

    def calc_atr(self, days=20):

        TR = self.calc_tr()  # These the true ranges for today.
        ATR = TR.rolling(days).mean()  # These are the average TR
        return ATR

    def calc_tr(self):
        # Calculate true range
        df = pd.DataFrame({'A1': np.abs(self.data.High - self.data.Close.shift(1)),
                           'A2': np.abs(self.data.Low - self.data.Close.shift(1)),
                           'A3': self.data.High - self.data.Low}).max(axis=1, skipna=True)
        return df

    def calc_Chandelier(self, fac=3, inplace=True):
        dat = self.data
        ATR = self.calc_atr(22)
        MAXP = dat["High"].rolling(window=22, center=False).max()
        LOWP = dat["Low"].rolling(window=22, center=False).min()

        Long = MAXP - ATR * fac
        Short = LOWP + ATR * fac
        return (Long, Short)

    def make_percentage_higher(self, days=10, min_shift=0.01):
        dat = self.data
        # Reduce the noise
        temp = dat.Close
        rat = ((temp.shift(1) / temp) > (1+min_shift))*1
        return rat.rolling(days).mean()

    @property
    def days(self):
        return self._days

    @days.setter
    def days(self, day):
        self._days = day
        self._long_predictors = self.make_long()

    @property
    def props(self):
        return pd.concat([self._shortPredictors, self._long_predictors], axis=1).dropna()

    @props.setter
    def props(self, properties):
        self.data = properties
        self._shortPredictors = self.make_short()
        self._long_predictors = self.make_long()

    def __str__(self):
        return self.data.__str__()
