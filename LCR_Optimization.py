pip install git+https://github.com/NaveenMSE/pymarkowitz

from pymarkowitz import *

import numpy as np
import pandas as pd

df = pd.read_excel('/content/test_data_hck.xlsx',sheet_name='Final_Data')

print(df.head())

pivot_df = df.pivot_table(index='Date', columns='Stock', values='Amount')
print(pivot_df.head())

# prompt: find the daily return for each stock above and take average

daily_returns = pivot_df.pct_change()
average_returns = daily_returns.mean(axis=0)
print(average_returns)

mu=list(average_returns)
cov_matrix=daily_returns.cov()
print(cov_matrix)

# prompt: put level 1 assets stock name in a list named level1_assets, similarly for level2A and level2B

level1_assets = list(df[df['Level '] == 1]['Stock'])
level2A_assets = list(df[df['Level '] == 2]['Stock'])
level2B_assets = list(df[df['Level '] == 3]['Stock'])

# prompt: take the stock names from the stock column and store it in the list as name assets
assets= list(df['Stock'].unique())

# prompt: ctreate a data frane security_data with the security name unique, its level and amount

security_data = df.groupby(['Stock', 'Level ']).agg(Amount=pd.NamedAgg(column='Amount', aggfunc='sum')).reset_index()
security_data

security_data = pd.DataFrame({
    'Security': security_data['Stock'].unique(),
    'Category':security_data['Level '],
    'amount': security_data['Amount']
})
security_data

# prompt: but the below caegory percentagees as dictionary and store it in level_allocations

level_allocations = {}
for category in security_data['Category'].unique():
  level_allocations[category] = round((security_data[security_data['Category'] == category]['amount'].sum() / security_data['amount'].sum()),2)
level_allocations

PortOpt=Optimizer(mu,cov_matrix)
# Define the custom constraint function
def level_allocation_const(level_allocations, security_data, assets):
    constraints = []

    for level, allocation in level_allocations.items():
        def level_constraint(w, level=level, allocation=allocation):
            # Initialize category weights
            category_weights = {category: 0 for category in level_allocations.keys()}

            # Calculate category weights
            for _, row in security_data.iterrows():
                category = row['Category']
                security = row['Security']
                if security in assets:
                    asset_index = assets.index(security)
                    category_weights[category] += w[asset_index]

            # Calculate constraint for the current level
            return allocation - category_weights[level]

        # Append the inequality constraints for the current level
        constraints.append({"type": "ineq", "fun": level_constraint})
        constraints.append({"type": "ineq", "fun": lambda w, level=level, allocation=allocation: -level_constraint(w, level, allocation)})

    return constraints

constraints = level_allocation_const(level_allocations, security_data, assets)
print(constraints)


# Create constraints
constraints = level_allocation_const(level_allocations, security_data, assets)

# Add objective
PortOpt.add_objective("max_return")

# Add constraints to the optimizer
for i in range(len(constraints)):
  PortOpt.add_constraint(constraint_type="custom", constraint=[constraints[i]])

PortOpt.add_constraint("weight", weight_bound=(0,0.3), leverage=1)  # Portfolio Long/Short
PortOpt.add_constraint("concentration",top_holdings=3,top_concentration=0.8)

PortOpt.solve()

PortOpt.summary()

PortOpt.constraint_options()

PortOpt.objective_options()
