import matplotlib.pyplot as plt
import numpy as np

w=0.4
x = ["Manhattan","M_Filtered","M_Scaled","SVM","M-F-S","SVM_Filtered"]
AUC = [0.878,0.914,0.951,0.965,0.959,0.972]
EER = [0.18065765645731371,0.1484870487058271,0.11763692496206396,0.12054244703221502,0.10284572835503178,0.12557263803596627]

plt.bar(x,AUC,w,label="EER")

plt.xlabel("Models")
plt.ylabel("Values")
plt.title("Models Vs Values(Equal Error Rate)")
plt.legend()
plt.show()