library(tidyverse)
library(here)

# set_here("/Users/elisakreiss/Documents/business/projects/iterated_narration/study")

df = read_csv(here("data","01_pilot","01_psychStudents","raw.csv"))

# anonymous_df = select(df,-student_id)
# 
# write.csv(anonymous_df, file = here("data","01_pilot","01_psychStudents","raw.csv"), row.names = FALSE)

clean = df %>%
        select(chain,deadend,generation,reproduction,story_text,story_title)

clean$story_text = gsub("(.{201,}?)\\s", "\\1\n", clean$story_text)
clean$reproduction = gsub("(.{61,}?)\\s", "\\1\n", clean$reproduction)
write.csv(clean, file = here("data","01_pilot","01_psychStudents","flowchart_data.csv"), row.names = FALSE)
