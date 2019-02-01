library(tidyverse)
library(summarytools)
library(gridExtra)
library(corrplot)
library(here)

theme_set(theme_bw(18))

# df = read_csv(here("data","01_pilot","09_subjective-ratings","raw_confidential.csv"))
# 
# anonymous_df = df %>% 
#   filter(!is.na(worker_id)) %>% 
#   mutate(id=submission_id-min(submission_id)) %>% 
#   select(-worker_id,-assignment_id,-submission_id,-hit_id,-experiment_id)
# 
# write.csv(anonymous_df, file = here("data","01_pilot","09_subjective-ratings","raw.csv"), row.names = FALSE)

df = read_csv(here("data","01_pilot","09_subjective-ratings","raw.csv"))

glimpse(df)

# other info
df_subj = df %>% 
  select(id,age,gender,education,languages,enjoyment,timeSpent,comments) %>% 
  distinct() %>% 
  mutate_at(vars(languages), funs(str_to_lower(.))) %>% 
  mutate_at(vars(age), funs(as.integer(.)))

p_age = ggplot(df_subj,aes(x=age)) + 
  geom_bar(width = .5,
           fill = "orange") 

p_gen = ggplot(df_subj,aes(x=gender)) +
  geom_bar(width = .5,
           fill = "orange")

p_edu = ggplot(df_subj,aes(x=education)) +
  geom_bar(width = .5,
           fill = "orange")

p_lang = ggplot(df_subj,aes(x=languages)) +
  geom_bar(width = .5,
           fill = "orange")

p_enj = ggplot(df_subj,aes(x=enjoyment)) + 
  geom_bar(width = .5,
           fill = "orange")

p_time = ggplot(df_subj,aes(x=timeSpent)) +
  geom_histogram(fill = "orange")

grid.arrange(p_age,p_gen,p_edu,p_lang,p_enj,p_time)

unique(df_subj$comments)

# HitCorrect

# plots

df_clean = df %>%
  select(id,generation,chain,trial_type,question,slider_val,box_checked,story_reproduction,story_title,trial_number)

df_corr = df_clean %>% 
  mutate(response=ifelse(box_checked=="false",slider_val,NA)) %>% 
  select(id,trial_type,slider_val) %>% 
  group_by(id) %>% 
  spread(trial_type,slider_val) %>% 
  ungroup() %>% 
  mutate(pay_attention=ifelse(((control1_false>50) + (control2_false>50) + (control3_true<50) + (control4_true>50)) > 2,FALSE,TRUE)) %>% 
  filter(pay_attention==TRUE) %>%  
  select(-id,-control1_false,-control2_false,-control3_true,-control4_true, -pay_attention)

# chart.Correlation(df_corr, histogram=TRUE, pch=19)
res = cor(df_corr, use="complete.obs")
corrplot(res, method = "number", type = "upper", order = "FPC", 
         tl.col = "black", tl.srt = 45)

df.pca <- prcomp(df_corr, center = TRUE,scale. = TRUE)
df.pca$rotation

# condition split-up!

df_plots_slider = df %>% 
  filter(box_checked==FALSE) %>% 
  filter(generation==1)

p_slider = ggplot(df_plots_slider,aes(x=trial_type,y=slider_val)) +
  geom_point(alpha = 0.2,
             position = position_jitter(width = 0.1, height = 0),
             size = 2) + 
  # error bars 
  stat_summary(fun.data = "mean_cl_boot",
               geom = "linerange",
               color = "black",
               size = 1) + 
  # means
  stat_summary(fun.y = "mean",
               geom = "point",
               shape = 21,
               fill = "red",
               color = "black",
               size = 4) +
  facet_grid(rows = vars(story_title)) +
  theme(axis.text.x = element_text(angle = 45))

# TODO: needs to be separated by generation
df_plots_box = df %>% 
  # filter(box_checked=="true") %>% 
  select(box_checked,trial_type) %>% 
  group_by(trial_type,box_checked) %>% 
  summarize(total_count = n()) %>%
  spread(key=box_checked,value=total_count)  

p_checkbox = ggplot(df_plots_box,aes(x=trial_type,y=true)) +
  geom_bar(stat = "identity",
           fill = "orange") +
  facet_grid(rows = vars(generation)) +
  theme(axis.text.x = element_text(angle = 45))

grid.arrange(p_slider,p_checkbox)
