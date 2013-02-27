# パッケージのインストールとライブラリ呼び出し
library('mi')

filename_in <- commandArgs()[4]
filename_out <- commandArgs()[5]

# データの読み込みと不要変数の削除
train <- read.table(filename_in, header=T, sep=',')
train.mi <- train

# NAデータのプロット
mp.plot(train.mi, clustered=FALSE) 

# データの情報取得と確認
train.info <- mi.info(train.mi)
train.info$imp.formula

# 欠損値補完の前処理と確認
train.pre <- mi.preprocess(train.mi)
attr(train.pre, 'mi.info')

# 欠損値補完とデータの取得
train.imp <- mi(train.pre, R.hat=1.8)
train.dat.all <- mi.completed(train.imp)

#
mi.data.frame(train.imp, m=1)

#　データの出力
 write.table(mi.data.frame(train.imp, m=1), filename_out, quote=F, col.names=T, append=T,sep = ",")
