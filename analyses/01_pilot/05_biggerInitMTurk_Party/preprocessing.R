library(tidyverse)
library(here)

# set_here("/Users/elisakreiss/Documents/business/projects/iterated_narration")

# df = read_csv(here("data","01_pilot","05_biggerInitMTurk_Party","raw_confidential.csv"))
# 
# anonymous_df = select(df,-workerId)
# anonymous_df$submission_id = anonymous_df$submission_id - min(unique(anonymous_df$submission_id)) + 1
# 
# write.csv(anonymous_df, file = here("data","01_pilot","05_biggerInitMTurk_Party","raw.csv"), row.names = FALSE)

df = read_csv(here("data","01_pilot","05_biggerInitMTurk_Party","raw.csv"))

clean = df %>%
        select(chain,deadend,generation,reproduction,story_text,story_title)

clean$story_text = gsub("(.{201,}?)\\s", "\\1\n", clean$story_text)
clean$reproduction = gsub("(.{61,}?)\\s", "\\1\n", clean$reproduction)
write.csv(clean, file = here("data","01_pilot","05_biggerInitMTurk_Party","flowchart_data.csv"), row.names = FALSE)
