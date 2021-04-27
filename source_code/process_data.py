import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 싱글톤 패턴
class Settings:
    """
    전역 설정 클래스

    Attributes:
        csv_path: .csv 파일이 위치한 경로(str)
    """
    def __new__(cls):
        if not hasattr(cls, "_instance"):
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        cls = type(self)
        if not hasattr(cls, "_init"):
            cls._init = True

            self.csv_path = "./Data/continuous_factory_process.csv"

    def set_csv_path(self, path):
        """
        .csv 파일의 경로 재설정

        Args:
            path: .csv 파일의 경로(str)
        """
        self.csv_path = path


class Data:
    """
    기초 전처리 클래스(열 분리)
    
    Attributes:
        factory_condition: Factory ambient conditions
        machine1_raw: First stage, Machine 1, raw material properties (material going in to Machine 1)
        machine1_process: First stage, Machine 1 process variables
        machine2_raw: First stage, Machine 2, raw material properties (material going in to Machine 2)
        machine2_process: First stage, Machine 2 process variables
        machine3_raw: First stage, Machine 3, raw material properties (material going in to Machine 3)
        machine3_process: First stage, Machine 3 process variables
        combine_params: Combiner stage process parameters. Here we combines the outputs from Machines 1, 2, and 3.
        primary_output: PRIMARY OUTPUT TO CONTROL: Measurements of 15 features (in mm), along with setpoint or target for each
        machine4_process: Second stage, Machine 4 process variables
        machine5_process: Second stage, Machine 5 process variables
        secondary_output: SECONDARY OUTPUT TO CONTROL: Measurements of 15 features (in mm), along with setpoint or target for each
        primary_output_actual: actual in SECONDARY OUTPUT TO CONTROL
        primary_output_setpoint: setpoint in SECONDARY OUTPUT TO CONTROL
    """
    def __init__(self):
        self.data_all = pd.read_csv(Settings().csv_path)
        self.factory_condition = self.data_all.iloc[:,1:3]
        self.machine1_raw = self.data_all.iloc[:,3:7]
        self.machine1_process = self.data_all.iloc[:,7:15]
        self.machine2_raw = self.data_all.iloc[:,15:19]
        self.machine2_process = self.data_all.iloc[:,19:27]
        self.machine3_raw = self.data_all.iloc[:,27:31]
        self.machine3_process = self.data_all.iloc[:,31:38]
        self.combine_params = self.data_all.iloc[:,39:42]
        self.primary_output = self.data_all.iloc[:,42:72]
        self.machine4_process = self.data_all.iloc[:,72:79]
        self.machine5_process = self.data_all.iloc[:,79:86]
        self.secondary_output = self.data_all.iloc[:,86:]
        actual_index = [i * 2 for i in range(15)]
        setpoint_index = [i * 2 + 1 for i in range(15)]
        self.primary_output_actual = self.primary_output.iloc[:,actual_index]
        self.primary_output_setpoint = self.primary_output.iloc[:,setpoint_index]
            
