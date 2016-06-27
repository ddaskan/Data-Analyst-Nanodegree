library("psych", lib.loc="~/R/win-library/3.2")
df=read.csv('stroopdata.csv')
df.describe()
summary(df)
describe(df)

#scatter
plot(df, col = "red", lwd=8,xlab="Time for congruent list (s)",ylab="Time for incongruent list (s)")
title("Scatter plot for the subjects in sample sets")

#boxplot
boxplot(df, col = "red", lwd=1.5,ylab = "The time it takes to name the ink colors (s)")
title("Box plot for the subjects in sample sets")

#line graph
matplot(df, type = c("p"),pch=1,col=1:2,xlab = "Subject No.", 
        ylab = "The time it takes to name the ink colors (s)",lwd=3)
legend("topleft", legend =c("Congruent","Incongruent"), col=1:2, pch=1)
title("Line graph of sample sets showing times to name the ink colors")

#Standardized
standf=scale(df)
matplot(standf, type = c("p"),pch=1,col=1:2,xlab = "Subject No.", 
        ylab = "Standardized times to name the ink colors",lwd=3)
legend("topleft", legend =c("Congruent","Incongruent"), col=1:2, pch=1)
title("Standardized line graph of sample sets showing times to name the ink colors")

#t-test
t.test(df$Incongruent,df$Congruent, paired=T)

