df=read.csv('titanic_data.csv')
t.test(df$Survived[df$Sex=='male']~df$Survived[df$Sex=='female'])



