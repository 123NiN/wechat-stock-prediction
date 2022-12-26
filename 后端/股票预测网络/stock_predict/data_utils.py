import tushare as ts
import pandas as pd
import os

# ----------------------下载某只股票数据------------------- #
# code:股票编码 日期格式：2019-05-21 filename：写到要存放数据的根目录即可如D:\data\
# length是筛选股票长度，默认值为False，既不做筛选，可人为指定长度，如200，既少于200天的股票不保存
def get_stock_data(code, date1, date2, filename, length=-1):
	df = ts.get_hist_data(code, start=date1, end=date2)
	df1 = pd.DataFrame(df)
	df1 = df1[['open', 'high', 'close', 'low', 'volume', 'p_change']]
	df1 = df1.reindex(index=df1.index[::-1])
	print('共有%s天数据' % len(df1))
	if length == -1:
		path = code + '.csv'
		df1.to_csv(os.path.join(filename, path))
	else:
		if len(df1) >= length:
			path = code + '.csv'
			df1.to_csv(os.path.join(filename, path))

# 辅助函数
def quchong(file):
	f = open(file)
	df = pd.read_csv(f, header=0)
	datalist = df.drop_duplicates()
	datalist.to_csv(file)


if __name__ == '__main__':
	get_stock_data('600050', '2019-01-1', '2021-08-23', '.\data\\')
