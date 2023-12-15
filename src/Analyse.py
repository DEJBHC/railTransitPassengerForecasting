from holoviews import opts
from pyecharts.globals import ThemeType
from wordcloud import WordCloud, ImageColorGenerator
from pyecharts.charts import Line, Bar, Geo
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import jieba
import seaborn as sns
from pyecharts import options as opts
# 设置列名与数据对齐
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)
# 显示10行
pd.set_option('display.max_rows', 10)
# 读取数据
df = pd.read_csv('../res/subway_all.csv', header=None, names=['city', 'line', 'name'])

# 各个城市地铁线路情况
df_line = df.groupby(['city', 'line']).count().reset_index()
print(df_line)

def create_map(df):
    # 绘制地图
    value = [i for i in df['line']]
    attr = [i for i in df['city']]
    data=[]
    for i in range(len(value)):
        if attr[i]=='city':
            continue
        else:
            data.append([attr[i],value[i]])
    print(data)
    print("*******************")
    geo=Geo()
    geo.add_schema(maptype="china",label_opts=opts.LabelOpts(is_show=False))
    geo.add(series_name="geo",data_pair=data,type_="effectScatter",symbol_size=1,color="orange",)
    geo.set_global_opts(title_opts=opts.TitleOpts(title="已开通地铁城市分布情况"),legend_opts=opts.LegendOpts(is_show=True))
    geo.render("../res/已开通地铁城市分布情况.html")


def create_line(df):
    """
    生成城市地铁线路数量分布情况
    """
    title_len = df['line']
    bins = [0, 5, 10, 15, 20, 25]
    level = ['0-5', '5-10', '10-15', '15-20', '20以上']
    len_stage = pd.cut(title_len, bins=bins, labels=level).value_counts().sort_index()
    len_stage.plot(kind='bar',x=0,y=1,color='red',title='各城市地铁线路数量分布')
     # 汉字字体，优先使用楷体，找不到则使用黑体
    plt.rcParams['font.sans-serif'] = ['Kaitt', 'SimHei']
    # 正常显示负号
    plt.rcParams['axes.unicode_minus'] = False
    plt.savefig('../res/各城市地铁线路数量分布')
    plt.show()


# 各个城市地铁线路数
print('各个城市地铁线路数')
df_city = df_line.groupby(['city']).count().reset_index().sort_values(by='line', ascending=False)
print(df_city)
create_map(df_city)
create_line(df_city)

# 哪个城市哪条线路地铁站最多
print('哪个城市哪条线路地铁站最多')
print(df_line.sort_values(by='name', ascending=False))

# 去除重复换乘站的地铁数据
print('去重后的地铁数据')
df_station = df.groupby(['city', 'name']).count().reset_index()
print(df_station)

# 统计每个城市包含地铁站数(已去除重复换乘站)
print('每个城市包含地铁站数')
print(df_station.groupby(['city']).count().reset_index().sort_values(by='name', ascending=False))


def create_wordcloud(df):
    """
    生成地铁名词云
    """
    # 分词
    text = ''
    for line in df['name']:
        text += ' '.join(jieba.cut(line, cut_all=False))
        text += ' '
    backgroud_Image = plt.imread('../res/tree2.jpg')
    wc = WordCloud(
        background_color='white',
        mask=backgroud_Image,
        font_path='STXINGKA.TTF',
        max_words=1000,
        max_font_size=150,
        min_font_size=15,
        prefer_horizontal=1,
        random_state=50,
    )
    wc.generate_from_text(text)
    img_colors = ImageColorGenerator(backgroud_Image)
    wc.recolor(color_func=img_colors)
    # 看看词频高的有哪些
    process_word = WordCloud.process_text(wc, text)
    sort = sorted(process_word.items(), key=lambda e: e[1], reverse=True)
    print(sort[:50])
    plt.imshow(wc)
    plt.axis('off')
    plt.show()
    wc.to_file("../res/地铁名词云.jpg")
    print('生成词云成功!')


create_wordcloud(df_station)

words = []

for line in df['name']:
    for i in line:
    # 将字符串输出一个个中文
        words.append(i)


def all_np(arr):
    """
    统计单字频率
    """
    arr = np.array(arr)
    key = np.unique(arr)
    result = {}
    for k in key:
        mask = (arr == k)
        arr_new = arr[mask]
        v = arr_new.size
        result[k] = v
    return result


def create_word(word_message):
    """
    生成柱状图
    """
    attr = [j[0] for j in word_message]
    v1 = [j[1] for j in word_message]
    bar = (
        Bar()
        .add_xaxis(attr)
        .add_yaxis("项目", v1)
        .set_global_opts(
            title_opts=opts.TitleOpts(title="中国地铁站最爱用的字"),
            yaxis_opts=opts.AxisOpts(name="数量"),
            xaxis_opts=opts.AxisOpts(name="项目"),)
    )
    # bar = Bar("中国地铁站最爱用的字", title_pos='center', title_top='18', width=800, height=400)
    # bar.add("", attr, v1, is_stack=True, is_label_show=True)
    bar.render("../res/中国地铁站最爱用的字.html")


word = all_np(words)
word_message = sorted(word.items(), key=lambda x: x[1], reverse=True)[:10]
create_word(word_message)

# 选取上海的地铁站
df1 = df_station[df_station['city'] == '上海']
print(df1)
# 选取上海地铁站名字包含路的数据
df2 = df1[df1['name'].str.contains('路')]
print('上海地铁站名字包含路的数据')
print(df2)

# 选取武汉的地铁站
df1 = df_station[df_station['city'] == '武汉']
print(df1)
# 选取武汉地铁站名字包含家的数据
df2 = df1[df1['name'].str.contains('家')]
print('武汉地铁站名字包含家的数据')
print(df2)

# 选取重庆的地铁站
df1 = df_station[df_station['city'] == '重庆']
print(df1)
# 选取重庆地铁站名字包含家的数据
df2 = df1[df1['name'].str.contains('家')]
print('重庆地铁站名字包含家的数据')
print(df2)

# 选取哈尔滨的地铁站
df1 = df_station[df_station['city'] == '哈尔滨']
print(df1)
# 选取哈尔滨地铁站名字包含家的数据
df2 = df1[df1['name'].str.contains('路')]
print('哈尔滨地铁站名字包含路的数据')
print(df2)
# 选取哈尔滨的地铁站
df1 = df_station[df_station['city'] == '哈尔滨']
print(df1)
# 选取哈尔滨地铁站名字包含家的数据
df2 = df1[df1['name'].str.contains('街')]
print('哈尔滨地铁站名字包含街的数据')
print(df2)


def create_door(door):
    """
    生成柱状图
    """
    attr = [j for j in door['city'][:3]]
    v1 = [j for j in door['line'][:3]]
    bar = (
        Bar()
        .add_xaxis(attr)
        .add_yaxis("项目", v1)
        .set_global_opts(
            title_opts=opts.TitleOpts(title="地铁站最爱用“门”命名的城市"),
            yaxis_opts=opts.AxisOpts(name="数量"),
            xaxis_opts=opts.AxisOpts(name="项目"),)
    )
    # bar = Bar("地铁站最爱用“门”命名的城市", title_pos='center', title_top='18', width=800, height=400)
    # bar.add("", attr, v1, is_stack=True, is_label_show=True, yaxis_max=40)
    bar.render("../res/地铁站最爱用门命名的城市.html")


# 选取地铁站名字包含门的数据
df1 = df_station[df_station['name'].str.contains('门')]
# 对数据进行分组计数
df2 = df1.groupby(['city']).count().reset_index().sort_values(by='line', ascending=False)
print('地铁站名字包含门的数据')
print(df2)
create_door(df2)


# 选取北京的地铁站
df1 = df_station[df_station['city'] == '北京']
print(df1)
# 选取北京地铁站名字包含门的数据
print('北京地铁站名字包含门的数据')
df2 = df1[df1['name'].str.contains('门')]
print(df2)

# 选取南京的地铁站
df1 = df_station[df_station['city'] == '南京']
# 选取南京地铁站名字包含门的数据
df2 = df1[df1['name'].str.contains('门')]
print('南京地铁站名字包含门的数据')
print(df2)

# 选取西安的地铁站
df1 = df_station[df_station['city'] == '西安']
# 选取西安地铁站名字包含门的数据
df2 = df1[df1['name'].str.contains('门')]
print('西安地铁站名字包含门的数据')
print(df2)

#选取数量前5个名字中带有大学的地铁站的城市，并绘制柱状图
df1=df[df['name'].str.contains('大学')]
city_counts=df1['city'].value_counts()
plt.figure(figsize=(10,5))
labelline=list(city_counts[:5].index)#
print(labelline)#['上海', '沈阳', '北京', '天津', '重庆']
plt.xlabel('城市')
plt.ylabel('站点数量')
plt.title('名字中带有大学的地铁站的城市数量分布')
plt.bar([i for i in labelline],city_counts[:5])

# 汉字字体，优先使用楷体，找不到则使用黑体
plt.rcParams['font.sans-serif'] = ['Kaitt', 'SimHei']
# 正常显示负号
plt.rcParams['axes.unicode_minus'] = False
plt.show()
plt.savefig('../res/名字中带有大学的地铁站的城市数量分布')


#绘制北京、武汉、天津、上海等各线路站点数量的折线图趋势分布
#北京：
df1=df[df['city']=='北京']
Bei_station=df1['line'].value_counts()
print(Bei_station)
plt.figure(figsize=(12,6))
labelline=list(Bei_station[:8].index)
plt.xlabel=('线路')
plt.ylabel=('各站点数量')
plt.title("北京各线路站点数量的分布趋势")
plt.plot([i for i in labelline],Bei_station[:8])
plt.savefig('../res/北京各线路站点数量的分布趋势')
plt.show()


#武汉
df1=df[df['city']=='武汉']
Wu_station=df1['line'].value_counts()
print(Wu_station)
plt.figure(figsize=(12,6))
labelline=list(Wu_station[:8].index)
plt.xlabel=('线路')
plt.ylabel=('各站点数量')
plt.title("武汉各线路站点数量的分布趋势")
plt.plot([i for i in labelline],Wu_station[:8])
plt.savefig('../res/武汉各线路站点数量的分布趋势')
plt.show()

#天津
df1=df[df['city']=='天津']
Tian_station=df1['line'].value_counts()
print(Tian_station)
plt.figure(figsize=(12,6))
labelline=list(Tian_station[:8].index)
plt.xlabel=('线路')
plt.ylabel=('各站点数量')
plt.title("天津各线路站点数量的分布趋势")
plt.plot([i for i in labelline],Tian_station[:8])
plt.savefig('../res/天津各线路站点数量的分布趋势')
plt.show()

#上海
df1=df[df['city']=='上海']
Shang_station=df1['line'].value_counts()
print(Shang_station)
plt.figure(figsize=(12,6))
labelline=list(Shang_station[:8].index)
plt.xlabel=('线路')
plt.ylabel=('各站点数量')
plt.title("上海各线路站点数量的分布趋势")
plt.plot([i for i in labelline],Shang_station[:8])
plt.savefig('../res/上海各线路站点数量的分布趋势')
plt.show()

#哈尔滨
df1=df[df['city']=='哈尔滨']
Ha_station=df1['line'].value_counts()
print(Ha_station)
plt.figure(figsize=(12,6))
labelline=list(Ha_station[:8].index)
plt.xlabel=('线路')
plt.ylabel=('各站点数量')
plt.title("哈尔滨各线路站点数量的分布趋势")
plt.plot([i for i in labelline],Ha_station[:8])
plt.savefig('../res/哈尔滨各线路站点数量的分布趋势')
plt.show()


#各个城市的线路数量的饼状图分布
line_count=df['city'].value_counts()
plt.figure(figsize=(10,7))
plt.pie(line_count,labels=line_count.index,autopct='%1.1f%%')
plt.title('各个城市的线路数量的饼状图分布')
plt.savefig('../res/各个城市的线路数量的饼状图分布')
plt.show()

#各个城市的站点数量的饼状图分布
#饼状图展示
df_station = df.groupby(['city', 'name']).count().reset_index()  #此处去除每个城市的重复换乘站点数，得到实际数量的站点数量
df1=df_station.groupby(['city']).count().reset_index().sort_values(by='name', ascending=False)
df1['city']=df1['city']+'(站点数'+df1['name'].map(str)+')'
line_count=df1['name']
plt.figure(figsize=(10,7))
plt.pie(line_count,labels=df1['city'],autopct='%1.1f%%')
plt.title('各个城市的站点数量的饼状图分布')
plt.savefig('../res/各个城市的站点数量的饼状图分布')
plt.show()

#散点图展示
df_station = df.groupby(['city', 'name']).count().reset_index()  #此处去除每个城市的重复换乘站点数，得到实际数量的站点数量
df1=df_station.groupby(['city']).count().reset_index().sort_values(by='name', ascending=False)
line_count=df1['name']
plt.figure(figsize=(10,7))
plt.xlabel=('城市')
plt.ylabel=('站点数量')
plt.scatter(x=df1['city'],y=line_count,marker='*')
plt.title('各个城市的站点数量的散点图分布')
plt.savefig('../res/各个城市的站点数量的散点图分布')
plt.show()

#各城市的每条线路的站点数量的变化 折线图
df1=df_line.sort_values(by='name', ascending=False)#by中指定按照什么列排序，ascending中默认升序排列，值为True
station_count=df1['line']+df1['city']
plt.figure(figsize=(15,8))
labelline=list(station_count[:12])
plt.xlabel=('线路')
plt.ylabel=('各站点数量')
plt.title("各城市各线路的站点数量前10的变化")
plt.plot([i for i in labelline],df1['name'][:12])
plt.savefig('../res/各城市各线路的站点数量前10的变化')
plt.show()

#每个城市的哪条线路的地铁站点数量最多  柱形图
df_1=df_line.sort_values(by='name', ascending=False)
df_2=df_1.groupby('city')['name'].max().reset_index(drop=False)#保留索引
line_station_c=df_2.sort_values(by='name',ascending=False)
# line_station_c.to_csv("../1.csv",header=False,index=False)
plt.figure(figsize=(15,5))
labelline=list(line_station_c['city'])
# line_text=pd.merge(left=line_station_c,right=df_1,on=['city','station'],how='inner')
# line_text.to_csv("../2.csv",header=False,index=False)
labelline=labelline#+line_text['line'].map(str)
plt.xlabel=('城市')
plt.ylabel=('站点数量')
plt.bar([i for i in labelline],line_station_c['name'])
plt.title('每个城市哪条线路的站点数最多')
plt.savefig('../res/每个城市哪条线路的站点数最多')
plt.show()

#统计各个城市的大学数量，然后利用回归图进行拟合（分析各个城市的大学数量与站点数量的关系
df_uni= pd.read_csv('../res/university.csv', header=None, names=['city', 'uni_count'], encoding='gbk')
df_uni=pd.merge(left=line_station_c,right=df_uni,on='city',how='inner') #将两个表格中的数据基于city列进行内连接。
x=df_uni['uni_count']
y=df_uni['name']
sns.regplot(x=x,y=y,color='b')
plt.title('分析各个城市的大学数量与站点数量的关系')
plt.savefig('../res/分析各个城市的大学数量与站点数量的关系')
plt.show()

#散点图
fig=plt.figure(figsize=(10,7))
plt.xlabel=('站点数量')
plt.ylabel=('大学数量')
plt.title('各个城市的大学数量与站点数量的关系')
plt.scatter(x=x,y=y,cmap='b',marker='*',alpha=0.8)
plt.grid()
plt.savefig('../res/各个城市的大学数量与站点数量的关系')
plt.show()

#seaborn的双变量图：可以查看多变量之间的分布关系，也可以显示它本身的单变量情况

df_s=df_uni
sns.jointplot(x='uni_count',y='name',data=df_s)
plt.savefig('../res/大学数量与站点数量的双变量图')
plt.show()
plt.close()

#选取郑州、武汉、广州、长沙同名的线路1-线路6，绘制折线图分析这些城市的目标线路的站点数量分布
df_1=df_line.sort_values(by='name', ascending=False)
zz_=df_1[df_1['city']=='郑州'].sort_values(by='line',ascending=False).reset_index()#ascending参数值为False时，则数据按指定列降序排序。
zz_=zz_.loc[zz_['line'].isin(['1号线','2号线','3号线','4号线','5号线','6号线'])]

wh_=df_1[df_1['city']=='武汉'].sort_values(by='line',ascending=False).reset_index()
wh_=wh_.loc[wh_['line'].isin(['1号线','2号线','3号线','4号线','5号线','6号线'])]

gz_=df_1[df_1['city']=='广州'].sort_values(by='line',ascending=False).reset_index()
gz_=gz_.loc[gz_['line'].isin(['1号线','2号线','3号线','4号线','5号线','6号线'])]

cs_=df_1[df_1['city']=='长沙'].sort_values(by='line',ascending=False).reset_index()
cs_=cs_.loc[cs_['line'].isin(['1号线','2号线','3号线','4号线','5号线','6号线'])]

print(zz_)
print(wh_)
print(gz_)
print(cs_)
plt.figure(figsize=(10,7))
L1=plt.plot(zz_['line'],zz_['name'],color='b',label='郑州线路1-6的站点数量变化')
L2=plt.plot(wh_['line'],wh_['name'],color='g',label='武汉线路1-6的站点数量变化')
L3=plt.plot(gz_['line'],gz_['name'],color='r',label='广州线路1-6的站点数量变化')
L4=plt.plot(cs_['line'],cs_['name'],color='k',label='长沙线路1-6的站点数量变化')
plt.legend()
plt.title('郑州、武汉、广州、长沙同名的线路1-线路6的站点数量分布')
plt.xlabel=('线路1-线路6')
plt.ylabel=('站点数量')
plt.savefig('../res/郑州、武汉、广州、长沙同名的线路1-线路6的站点数量分布')
plt.show()

#选取广州、天津、武汉、重庆同名的线路1-线路6，绘制折线图分析这些城市的目标线路的站点数量分布
df_1=df_line.sort_values(by='name', ascending=False)
zz_=df_1[df_1['city']=='广州'].sort_values(by='line',ascending=False).reset_index()#ascending参数值为False时，则数据按指定列降序排序。
zz_=zz_.loc[zz_['line'].isin(['1号线','2号线','3号线','4号线','5号线','6号线'])]

wh_=df_1[df_1['city']=='天津'].sort_values(by='line',ascending=False).reset_index()
wh_=wh_.loc[wh_['line'].isin(['1号线','2号线','3号线','4号线','5号线','6号线'])]

gz_=df_1[df_1['city']=='武汉'].sort_values(by='line',ascending=False).reset_index()
gz_=gz_.loc[gz_['line'].isin(['1号线','2号线','3号线','4号线','5号线','6号线'])]

cs_=df_1[df_1['city']=='重庆'].sort_values(by='line',ascending=False).reset_index()
cs_=cs_.loc[cs_['line'].isin(['1号线','2号线','3号线','4号线','5号线','6号线'])]

print(zz_)
print(wh_)
print(gz_)
print(cs_)
plt.figure(figsize=(10,7))
L1=plt.plot(zz_['line'],zz_['name'],color='b',label='广州线路1-6的站点数量变化')
L2=plt.plot(wh_['line'],wh_['name'],color='g',label='天津线路1-6的站点数量变化')
L3=plt.plot(gz_['line'],gz_['name'],color='r',label='武汉线路1-6的站点数量变化')
L4=plt.plot(cs_['line'],cs_['name'],color='k',label='重庆线路1-6的站点数量变化')
plt.legend()
plt.title('广州、天津、武汉、重庆同名的线路1-线路6的站点数量分布')
plt.xlabel=('线路1-线路6')
plt.ylabel=('站点数量')
plt.savefig('../res/广州、天津、武汉、重庆同名的线路1-线路6的站点数量分布')
plt.show()

#全国各城市的总的换乘站点数量（2换乘、3换乘、4换乘等）分布统计
df_1=df.groupby(['city','name']).count().reset_index()
print(df_1)
df_1=df_1[df_1['line']>1]#筛选出来全国的换乘站点数
tran_sit=df_1.groupby('line').count().reset_index()  #保留原索引，但是值是count()函数计数之后的值
plt.figure(figsize=(10,5))
plt.xlabel=('站点可换乘等级')
plt.ylabel=('站点数量')
plt.bar(tran_sit['line'],tran_sit['name'],color='g')
plt.title('全国各城市总的换乘站点数量（2换乘、3换乘、4换乘等）分布统计')
plt.savefig('../res/全国各城市总的换乘站点数量（2换乘、3换乘、4换乘等）分布统计')
plt.show()
print(tran_sit[tran_sit['line']==5]['name'])


