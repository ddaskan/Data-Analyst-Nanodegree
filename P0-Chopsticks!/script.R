df=read.csv("chopstick-effectiveness.csv")
df$Chopstick.Length <-as.factor(df$Chopstick.Length)
m1=aov(Food.Pinching.Efficiency~Chopstick.Length, data=df)
summary(m1)
t.test(df$Food.Pinching.Efficiency[df$Chopstick.Length=="240"],
       df$Food.Pinching.Efficiency[df$Chopstick.Length=="330"])

