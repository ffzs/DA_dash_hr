import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import random

def make_dash_table(df):
    table = []
    html_col_name = ['-']
    for col in df.columns:
        html_col_name.append(html.Td([col]))
    table.append(html.Tr(html_col_name))
    for index, row in df.iterrows():
        html_row = [index]
        for i in range(len(row)):
            html_row.append(html.Td([row[i]]))
        table.append(html.Tr(html_row))
    return table

def make_table_without_col(df):
    table = []
    for index, row in df.iterrows():
        html_row = [index]
        for i in range(len(row)):
            html_row.append(html.Td([row[i]]))
        table.append(html.Tr(html_row))
    return table

# data perpare
df_o = pd.read_csv('data/HR_comma_sep.csv')
df = df_o.copy()
dfi = df_o.copy()
df_data_description = pd.read_csv('data/data_description.csv')
df_cluster = pd.read_csv('data/hr_cluster.csv')
knn_scores = pd.read_csv('data/knn_scores.csv')
knn_scores = knn_scores.set_index('Unnamed: 0')

salary_dict = {"low":0, "medium":1, "high":2}
def obj2num_salary(salary):
    changer = salary_dict
    return changer[salary]

df['salary'] = df['salary'].apply(obj2num_salary)

sales_list = np.unique(df['sales'].values).tolist()
sales_dict, anti_sales_dict = {}, {}
for (item, index) in enumerate(sales_list):
    sales_dict[index] = item
    anti_sales_dict[item] = index
# 替换salsa的对象
def obj2num_sales(sales):
    changer = sales_dict
    return changer[sales]

df['sales'] = df['sales'].apply(obj2num_sales)
# 备份英文col
columns_en = df.columns
# 中文col
columns_zh = ['满意度水平', '绩效评估', '参加项目数', '平均每月工时', '工作年限', '是否有工作事故', '是否离职', '五年内是否晋升', '职业', '薪水']
# 更换中文col
df.columns = columns_zh
dfi.columns = columns_zh
# 复制带计数的df_c
df_c = dfi.copy()
df_c['计数'] = 1
# 计算变量之间关系
corr = df.corr(method='pearson').round(3)
# 离职人员dataframe
df_l = df.groupby('是否离职').get_group(1).drop(columns=['是否离职'])
# 离职人员相关性
corr_l = df_l.corr('pearson').round(3)
# 离职人员计数
# df_lc = df_c.groupby('是否离职').get_group(1).drop(columns=['是否离职'])

app = dash.Dash()

# page header
logo = html.Div([
        html.Div([
            html.Img(src='https://raw.githack.com/ffzs/DA_dash_hr/master/img/logo1.png', height='50', width='150')
        ], className="ten columns padded")
    ], className="row gs-header")

header = html.Div([
        html.Div([
            html.H5(
                '关于员工离职情况的数据分析报告')
        ], className="twelve columns padded")
    ], className="row gs-header gs-text-header")

# row 1
row1 = html.Div([
    html.Div([
        html.H6('数据背景',
                className="gs-header gs-text-header padded"),
        html.Br([]),
        dcc.Markdown('''数据来源于Kaggle网站，原文为"Our example concerns a big company that wants to understand why some of their 
               bestand most experienced employees are leaving prematurely. The company also wishes topredict which 
               valuable employees will leave next." ，就是想知道有经验的员工为什么过早离职，以及预测哪些有价值的员工将离职'''),
    ], className="twelve columns")
], className="row ")

# row2
row2 = html.Div([
    html.Div([
        html.H6(["数据说明"],
                className="gs-header gs-table-header padded"),
        html.Table(make_dash_table(df_data_description))
    ], className="five columns"),
    html.Div([
        html.H6(["数据剪影"],
                className="gs-header gs-table-header padded"),
        html.Table(make_dash_table(dfi.describe(include='all').round(2).fillna("")))
    ], className="seven columns"),
], className="row ")

# row3
pie_list = ['参加项目数', '工作年限', '是否有工作事故', '是否离职', '五年内是否晋升', '职业', '薪水']
type_list = ['数值', '占比']
row3 = html.Div([
    html.Div([
        html.H6(["各变量占比情况"],
                className="gs-header gs-table-header padded"),
        html.Div([
            dcc.Dropdown(
            id='pie_dropdown',
            options=[{'label':i, 'value':i} for i in pie_list],
            value=pie_list[3])
        ],style={'width': '25%','margin-left': '0%'}),
    dcc.Graph(id='pie')
    ], className="five columns"),
    html.Div([
        html.H6(["各变量分布情况"],
                className="gs-header gs-table-header padded"),
        html.Div([
            dcc.Dropdown(
                id='type_dropdown',
                options=[{'label': i, 'value': i} for i in type_list],
                value=type_list[0])
        ], style={'width': '15%', 'margin-left': '0%','display': 'inline-block'}),
        html.Div([
            dcc.Dropdown(
                id='bar_dropdown',
                options=[{'label': i, 'value': i} for i in columns_zh],
                value=columns_zh[3])
        ], style={'width': '20%', 'margin-left': '0%','display': 'inline-block'}),
        dcc.Graph(id='bar')
    ], className="seven columns"),
], className="row ")

# row4
row4 = html.Div([
    html.Div([
        html.H6(["全体相关性值热力图"],
                className="gs-header gs-table-header padded"),
        dcc.Graph(id='heatmap',
                  figure={
                      'data':[
                          go.Heatmap(z=np.fabs(corr.values),
                              x = corr.columns.tolist(),
                              y = corr.index.tolist(),
                              colorscale='YlGnBu')],
                      'layout':go.Layout(margin=dict(l=100, b=100, t=50))
                  })
    ], className="five columns"),
    html.Div([
        html.H6(["全体员工变量的皮尔森相关性"],
                className="gs-header gs-table-header padded"),
        html.Table(make_dash_table(corr)),
        html.H6(["全体员工数据分析"],
                className="gs-header gs-table-header padded"),
        html.Br([]),
        dcc.Markdown('''     
+ 离职率为23.8%，接近公司员工的四分之一
+ 每月工时的平均值为201小时，两个峰值区分别为150左右和260左右
+ 每月工作时间高于287小时的都离职了
+ 满意度低于0.11的都离职了
+ 离职员工工作年限集中于3-5年
+ 离职人员多为中低收入者，且少有工作失误
+ 有经验员工应为参加项目多员工,即参加项目数为5-7个的员工
+ 参加项目6-7个的员工有共性，满意度极低，工作时间长，绩效高，工资中下，未晋升，接近三分之二离职了，其中7个项目的全部离职
+ 根据变量相关性可知与离职相关性最强变量为满意度，其次是工资水平
+ 离职人员、有经验人员在各个职业分布比较均匀，即相关性不强
        '''),
    ], className="seven columns"),
], className="row ")

# row5
scatter3d_list0 = ['满意度水平', '工作年限', '五年内是否晋升', '是否有工作事故', '职业', '薪水']
scatter3d_list1 = ['工作年限', '五年内是否晋升', '是否有工作事故', '职业', '薪水']
employee_class = ['全体离职员工', '员工类1', '员工类2', '员工类3', '杂类']
cluster_list = ['K-means', 'DBSCAN']
p = pd.crosstab(df_cluster["计数"], df_cluster["km_labels"], margins=False)
q = pd.crosstab(df_cluster["计数"], df_cluster["db_labels"], margins=False)
cluster_table = p.append(q)
cluster_table.index=cluster_list
row5 = html.Div([
    html.Div([
        html.H6(["离职员工变量的皮尔森相关性"],
                className="gs-header gs-table-header padded"),
        html.Table(make_dash_table(corr_l)),
        html.H6(["离职员工相关性值热力图"],
                className="gs-header gs-table-header padded"),
        dcc.Graph(id='heatmap_l',
                  figure={
                      'data':[
                          go.Heatmap(z=np.fabs(corr_l.values),
                              x = corr_l.columns.tolist(),
                              y = corr_l.index.tolist(),
                              colorscale='YlGnBu')],
                      'layout':go.Layout(margin=dict(l=100, b=90, t=40))
                  })
        ], className="five columns"),
    html.Div([
        html.H6(["离职员工聚类情况"],
                className="gs-header gs-table-header padded"),
        html.Table(make_dash_table(cluster_table)),
        html.H6(["离职员工聚类3D散点图"],
                className="gs-header gs-table-header padded"),
        html.Div([
            dcc.Dropdown(
            id='cluster_dropdown',
            options=[{'label':i, 'value':i} for i in cluster_list],
            value=cluster_list[0]),
        ],style={'width': '20%','margin-left': '0%','display': 'inline-block'}),
        dcc.Graph(id='cluster'),
        dcc.Markdown('''
+ 分别使用DBSCAN、K-means聚类算法对本项目中离职员工的“满意度水平”、“绩效评估”、“平均每月工时”、“工作年限”、“参加项目数”五项变量做聚类处理
+ 对于本项目来说，DBSCAN相较K-means而言对杂项的区分能力更强
+ 上图中x轴为绩效评估，y轴为满意度水平，z轴为平均每月工时
        ''')
    ], className="seven columns"),
], className="row ")

# row6
row6 = html.Div([
    html.Div([
        html.H6(["离职员工数3d散点图"],
                className="gs-header gs-table-header padded"),
        html.Div([
            dcc.Dropdown(
                id='employee',
                options=[{'label': i, 'value': i} for i in employee_class],
                value=employee_class[0]),
        ], style={'width': '15%', 'margin-left': '0%', 'display': 'inline-block'}),
        html.Div([
            dcc.Dropdown(
                id='label0',
                options=[{'label': i, 'value': i} for i in scatter3d_list0],
                value=scatter3d_list0[0]),
        ], style={'width': '15%', 'margin-left': '0%', 'display': 'inline-block'}),
        html.Div([
            dcc.Dropdown(
                id='label1',
                options=[{'label': i, 'value': i} for i in scatter3d_list1],
                value=scatter3d_list1[0]),
        ], style={'width': '15%', 'margin-left': '0%', 'display': 'inline-block'}),
        html.Div([
            html.P('''
                x轴为绩效评估，y轴为每月工时，z轴为第一个选项框，颜色为第二个选项框，散点大小为参加项目数''')
        ], style={'width': '50%', 'margin-left': '3%', 'display': 'inline-block'}),
        dcc.Graph(id='scatter3d')
    ], className="seven columns"),
    html.Div([
        html.H6(["随机森林分类预测离职员工"],
                className="gs-header gs-table-header padded"),
        html.Br(),
        dcc.Markdown('''
+ 随机森林的预测准确率为98.57%
+ 查准率为99.43%
+ 查重率为94.74%
                '''),
        html.H6(["knn分类预测离职员工"],
                className="gs-header gs-table-header padded"),
        html.Table(make_table_without_col(knn_scores.round(2))),
        dcc.Graph(id='scatter',
                  figure={
                      'data': [
                          go.Scatter(
                              x=knn_scores.T.k.tolist(),
                              y=knn_scores.T.score.tolist(),
                              mode='markers'
                          )],
                      'layout': go.Layout(margin=dict(l=30, b=30, t=0, r=15), height=150)
                  }),
        html.Table(),
        dcc.Markdown('''
+ k为1时准确率最高，预测准确率为100%，故该模型k取1
+ k为1时查准率、查全率皆为100%
                '''),
    ], className="five columns"),
], className="row ")

# row7
row7 = html.Div([
    html.Div([
        html.H6(["离职员工数据分析"],
                className="gs-header gs-table-header padded"),
        html.Br([]),
        dcc.Markdown('''     
+ 根据离职员工数据的皮尔森相关性可知，他们的每月工时、绩效、工作年限、参加项目成强相关，同时与满意度成较强相关，与其他变量基本没有关系。
+ 第一条可以理解为就是说你工作是否努力，在公司工作了多少年，经验是否丰富，跟你开多少工资、有无升职机会没有任何关系
+ 根据3D散点图可以看出，离职员工有很明显的聚类现象，大部分离职人员可以被分为三类：
    1. 绩效低0.4左右，每月工作时长低140左右，满意度中下0.4左右，工作年限3年，参加项目3个
    2. 绩效高0.77-1，工时长240-310小时，满意度极低0.1左右，工作年限4-5年，参加项目6-7个
    3. 绩效高0.8-1，工时长210-270小时，满意度高0.7以上，工作年限5-6年，参加项目4-5个
+ 按聚类来看第一类员工1515左右人，第二类员工886人，第三类员工879人，第一类员工占比比较大
+ 第一类员工从工时绩效来看不算是优秀的员工，他们的离职对于公司来说可以接受
+ 第二类员工工作积极、经验丰富，满意度极低，容易识别，属于公司的有价值员工，离职很可惜
+ 第三类员工工作情况稍差于第二类，不过也算认真，满意度高，具有隐蔽性，属于有价值员工
                '''),

    ], className="seven columns"),
    html.Div([
        html.H6(["总结、思考、策略"],
                className="gs-header gs-table-header padded"),
        html.Br([]),
        dcc.Markdown('''
+ 总结：对于员工是否离职的预测knn和随机森林都有不错的表现都可以胜任，其中knn模型k为1时准确率为100%可能跟离职人员数据区域性明显，且重复数据多有关；对于离职人员的聚类
DBSCAN比K-means更适合。  
+ 思考：想要对公司员工的情况了解更加透彻的话仅仅是一年的数据还不够充分，最好要有连续几年的数据才能的高更将详尽的结果。   
+ 策略：第二类、第三类离职人员属于公司的优秀员工，他们的离职是公司的损失，hr应该尽早沟通才对，适当的调整工资或者给予其晋升机会；
将经验丰富人员的离职率归入管理人员的考核里可以适当减少管理人员与此类员工之间的摩擦
                '''),
    ], className="five columns"),
], className="row ")

# app layout
app.layout = html.Div([
    logo,
    header,
    html.Br([]),
    row1,
    row2,
    row3,
    row4,
    row5,
    row6,
    row7
], className="subpage")

@app.callback(
dash.dependencies.Output('pie', 'figure'),
[dash.dependencies.Input('pie_dropdown', 'value')])
def update_pie(value):
    count = df_c.groupby(value).count()
    trace = go.Pie(labels=count['计数'].index.tolist(), values=count['计数'].tolist())
    layout = go.Layout(margin=dict(t=0, b=0), height=400)
    return dict(data=[trace], layout=layout)

@app.callback(
dash.dependencies.Output('bar', 'figure'),
[dash.dependencies.Input('bar_dropdown', 'value'),
 dash.dependencies.Input('pie_dropdown', 'value'),
 dash.dependencies.Input('type_dropdown', 'value')])
def update_bar(value0, value1, type):
    cross = pd.crosstab(dfi[value0], dfi[value1], margins=True)
    cross_col_name = cross.columns.tolist()[:-1]
    cross_ = cross.copy()
    for name in cross_col_name:
        cross_[name] = cross_[name] / cross_['All']
    if type == '数值':
        cross_new = cross.iloc[:-1, :-1]
    else:
        cross_new = cross_.iloc[:-1, :-1]
    data = []
    for key in cross_new.columns.tolist():
        trace = go.Bar(
            x=cross_new.index.tolist(),
            y=cross_new[key].tolist(),
            name=key,
            opacity=0.6
        )
        data.append(trace)
    layout = go.Layout(barmode='stack', margin=dict(t=0, b=30),height=400)
    fig = go.Figure(data=data, layout=layout)
    return fig


@app.callback(
    dash.dependencies.Output('cluster','figure'),
    [dash.dependencies.Input('cluster_dropdown','value')]
)
def graph_cluster(value):
    changer = {'K-means':'km_labels','DBSCAN':'db_labels'}
    group_cluster = df_cluster.groupby(changer[value])
    data=[]
    for item in employee_class[1:]:
        dff = group_cluster.get_group(item)
        trace = go.Scatter3d(
            x=dff['绩效评估'],
            y=dff['满意度水平'],
            z=dff['平均每月工时'],
            name=item,
            mode='markers',
            marker=dict(
                size=4,
            ))
        data.append(trace)
    layout = go.Layout(margin=dict(l=20, r=20, t=0, b=30), height=450)
    fig = go.Figure(data=data, layout=layout)
    return fig

@app.callback(
    dash.dependencies.Output('scatter3d', 'figure'),
    [dash.dependencies.Input('employee','value'),
     dash.dependencies.Input('label0','value'),
     dash.dependencies.Input('label1','value')]
)
def update_scatter3d(employee_type, value0, value1):
    if len(set([value0, value1])) == 2:
        if employee_type == employee_class[0]:
            df_lc = df_cluster
        else:
            group_cluster = df_cluster.groupby('db_labels')
            df_lc = group_cluster.get_group(employee_type)
        df_left_pivot = pd.pivot_table(df_lc, values='计数', index=['绩效评估', '平均每月工时', value0, value1, '参加项目数'],
                                       aggfunc=np.count_nonzero)
        df_index = df_left_pivot.index.to_frame()
        data = []
        for i in np.unique(df_index[value1].values).tolist():
            dff = df_index[df_index[value1] == i]
            trace = go.Scatter3d(
                x=dff['绩效评估'],
                y=dff['平均每月工时'],
                z=dff[value0],
                text=dff['参加项目数'],
                name=i,
                mode='markers',
                marker=dict(
                    size=dff['参加项目数'] * 3,
                    sizemode='diameter',
                    sizeref=0,
                ))
            data.append(trace)
        layout = go.Layout(margin=dict(l=20, r=20, t=0, b=30),height=500)
        fig = go.Figure(data=data, layout=layout)
        return fig


external_css = ["https://cdnjs.cloudflare.com/ajax/libs/normalize/7.0.0/normalize.min.css",
                "https://cdnjs.cloudflare.com/ajax/libs/skeleton/2.0.4/skeleton.min.css",
                "//fonts.googleapis.com/css?family=Raleway:400,300,600",
                "http://raw.githack.com/ffzs/DA_dash_hr/master/css/my.css",
                "https://maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css"]

for css in external_css:
    app.css.append_css({"external_url": css})

external_js = ["https://code.jquery.com/jquery-3.2.1.min.js",
               "https://codepen.io/bcd/pen/YaXojL.js"]

for js in external_js:
    app.scripts.append_script({"external_url": js})

if __name__ == '__main__':
    app.run_server(debug=True)
