library(tidyverse)

df_1 = read_csv(here("data","02_main","02_subjective-ratings","raw_confidential1.csv"))
df_2 = read_csv(here("data","02_main","02_subjective-ratings","raw_confidential2.csv"))
df_3 = read_csv(here("data","02_main","02_subjective-ratings","raw_confidential3.csv"))

df = bind_rows(df_1,df_2,df_3)

anonymous_df = df %>%
  group_by(worker_id) %>% 
  mutate(anon_worker_id = ifelse(is.na(worker_id),
                                 NA,
                                 paste(sample(0:9,15,replace=TRUE),collapse = "")
                                 )) %>% 
  ungroup() %>% 
  select(-worker_id,-assignment_id,-submission_id,-hit_id,-experiment_id)

write.csv(anonymous_df, file = here("data","02_main","02_subjective-ratings","data.csv"), row.names = FALSE)
