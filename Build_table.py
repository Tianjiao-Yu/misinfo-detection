from prettytable import PrettyTable

table = PrettyTable()
table.field_names = ["Method", "Class", "Precision", "Recall", "F1-Score"]


table.add_row(["SVM-Ngram","T\nF",'0.797','0.882','0.837'])
table.add_row(["SVM_BOW","T\nF","0.889","0.904","0.896"])
table.add_row(['NB_Ngram',"T\nF",'0.803','0.987','0.886'])
table.add_row(["LG_Ngram","T\nF",'0.794','0.913',"0.850"])
table.add_row(['RandomForest',"T\nF",'0.797',"0.886",'0.839'])
table.add_row(['MLP_ngram',"T\nF",'0.806','1.0','0.892'])
table.add_row(['LSTM',"T\nF",'0.844','0.821','0.832'])
table.add_row(['HAN',"T\nF",'0.823','0.838','0.830'])
print(table)
