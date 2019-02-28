library(lme4)
library(tidyverse)
library(broom)

d1 = mtcars %>% 
  arrange(hp)
m1 = lm(mpg ~ cyl, data=d1)
summary(m1)
resid(m1)
glance(m1)


d2 = mtcars %>% 
  arrange(drat)
m2 = lm(mpg ~ cyl, data=d2)
summary(m2)
resid(m2)
glance(m2)=


d3 = mtcars %>% 
  arrange(cyl)
m3 = lm(mpg ~ cyl, data=d3)
summary(m3)
resid(m3)
glance(m3)


d4 = mtcars %>% 
  arrange(mpg)
m4 = lm(mpg ~ cyl, data=d4)
summary(m4)
resid(m4)
glance(m4)

mean(resid(m1))
mean(resid(m2))
mean(resid(m3))
mean(resid(m4))
