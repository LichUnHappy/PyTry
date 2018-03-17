import xlrd

book = xlrd.open_workbook('*.xlsx')
sheet = xlrd.sheet_by_name('* ')

data = {}

for i in range(sheet.nrows):
	row = sheet.row_values(i)

	keyattri = row[1]

	data[keyattri] = {
				   'attri1':{
				   'attri2':[row[4], row[5]],
				   'attri3':[row[6], row[7]],
				   			},
					 }