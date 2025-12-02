import pandas as pd



print("Q1:")
import pandas as pd
print(pd.__version__)

df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'Los Angeles', 'Chicago']
})
print(df, "\n")




print("Q2:")
S1 = pd.Series([100, 200, 300, 400, 500])
print(S1)
print(S1[1], S1[3])
S2 = pd.Series([10, 20, 30, 40, 50])
print(S1 + S2, "\n")




print("Q3:")
print(df[['Name', 'City']])
df['Salary'] = [50000, 60000, 70000]
print(df)
print(df['Age'].mean())
print(df['Salary'].sum(), "\n")




print("Q4:")
print(df[df['Age'] > 28])
df_indexed = df.set_index('Name')
print(df_indexed)
print(df_indexed.reset_index(), "\n")




print("Q5:")
data = {
    "Name": ["John", "Jane", "Emily"],
    "Department": ["Sales", "Marketing", "HR"],
    "Salary": [50000, 60000, 55000]
}
df_emp = pd.DataFrame(data)
print(df_emp)

filtered = df_emp[df_emp["Salary"] > 55000]
print(filtered[['Name', 'Department']], "\n")




print("Q6:")
print(df_emp.groupby('Department')['Salary'].mean())
print(df_emp.groupby('Department')['Salary'].agg(['min', 'max']), "\n")




print("Q7:")
df1 = pd.DataFrame({
    'Name': ['John', 'Jane', 'Emily'],
    'Department': ['Sales', 'Marketing', 'HR']
})
df2 = pd.DataFrame({
    'Name': ['John', 'Jane', 'Emily'],
    'Experience (Years)': [5, 7, 3]
})
merged = pd.merge(df1, df2, on='Name')
print(merged, "\n")




print("Q8:")
print(merged.sort_values(by='Experience (Years)', ascending=False))
