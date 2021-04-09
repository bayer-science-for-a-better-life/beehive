import pandas as pd
import plotly.express as px


methods = ['simple_thresholding', 'local_thresholding', 'simple_gradient',
           'double_gradient', 'refine_segmentation', 'refine_segmentation2']

results = []
for method in methods:
    df = pd.read_json(method + '/results.json')
    df['method'] = method
    results.append(df)
results = pd.concat(results)


fig = px.bar(
    results.melt(id_vars=['method', 'img'], value_vars=['PPV', 'TPR']),
    x='method', y='value', color='img', barmode='group', facet_row='variable',
    hover_data=['method', 'img', 'value', 'variable'])
fig.show()
fig.write_image('plots/comparison.png')
