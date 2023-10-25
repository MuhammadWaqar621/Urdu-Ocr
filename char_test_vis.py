
# First, create character-wise accuracy table in a CSV file by running ```char_test.py```
# Then visualize the result by running ```char_test_vis```

import pandas as pd
import matplotlib.pyplot as plt

# Read "Character-wise-accuracy.csv" with first row as header
df = pd.read_csv("Character-acc_HRNetDBiLSTM.csv", header=0)

# Insert characters you want to inspect
check_char = ['ا','آ', 'ب', 'پ', 'ت', 'ٹ',
              'ث', 'ج', 'چ', 'ح', 'خ',
              'د', 'ڈ', 'ذ', 'ر', 'ڑ',
              'ز', 'ژ', 'س', 'ش', 'ص',
              'ض', 'ط', 'ظ', 'ع', 'غ',
              'ف', 'ق', 'ک', 'ك', 'گ',
              'ل', 'م', 'ن', 'ں', 'و',
              'ہ', 'ھ', 'ء', 'ی', 'ے']

# Plot the accuracy of each character in check_char in a bar chart and saves it
df[df["Alphabet"].isin(check_char)].plot.bar(x="Alphabet", y="Accuracy", rot=0)
# df[df["Accuracy"]>=50].plot.bar(x="Alphabet", y="Accuracy", rot=0)
plt.savefig("Character-wise-accuracy.png")