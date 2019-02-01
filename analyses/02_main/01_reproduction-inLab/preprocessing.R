library(tidyverse)
library(here)

# set_here("/Users/elisakreiss/Documents/business/projects/iterated_narration")

# df = read_csv(here("data","02_main","01_reproduction-inLab","raw_confidential.csv"))
# 
# anonymous_df = arrange(df,submission_id) %>% 
#   mutate(id=sort(rep(seq.int(1,length(unique(submission_id))),5))) %>% 
#   select(-comments, -submission_id)
# 
# write.csv(anonymous_df, file = here("data","02_main","01_reproduction-inLab","raw.csv"), row.names = FALSE)

theme_set(theme_bw(18))

df = read_csv(here("data","02_main","01_reproduction-inLab","raw.csv"))

##################
# FLOWCHART DATA #
##################

d_flowchart = df %>%
        select(chain,deadend,generation,reproduction,story_text,story_title)

# there is only one deadend because of multiple entries
d_flowchart[d_flowchart$deadend=="true" & !d_flowchart$generation==5,]

d_flowchart$story_text = gsub("(.{201,}?)\\s", "\\1\n", d_flowchart$story_text)
d_flowchart$reproduction = gsub("(.{61,}?)\\s", "\\1\n", d_flowchart$reproduction)
# write.csv(d_flowchart, file = here("data","02_main","01_reproduction-inLab","flowchart_data.csv"), row.names = FALSE)

##########################
# Subjective Rating Data #
##########################

# seed stories
bees_free = "In late December 2017, a couple in Iowa was checking on their 50 beehives when they discovered a tragic scene. The hives had been overturned and hacked apart, and the equipment had been thrown out of the shed and smashed. This destruction caused the death of about half a million bees and approximately $60,000 in property damage. Nearly three weeks later, police arrested two boys (12 and 13 years old) who, allegedly, were responsible for the damage. The charges against them include criminal mischief, burglary, and offenses to an agricultural animal facility. Since they are still minors, they will be charged in juvenile court where they face up to 10 years in prison and fines of up to $10,000 if convicted. Police officials explained that the investigation is still in progress, and the evidence so far doesn't warrant rushed conclusions about the guilt of the suspects."
bees_jail = "In late December 2017, a couple in Iowa was checking on their 50 beehives when they discovered a tragic scene. The hives had been overturned and hacked apart, and the equipment had been thrown out of the shed and smashed. This destruction caused the death of about half a million bees and approximately $60,000 in property damage. Nearly three weeks later, police arrested two boys (12 and 13 years old) who, allegedly, were responsible for the damage. The charges against them include criminal mischief, burglary, and offenses to an agricultural animal facility. Since they are still minors, they will be charged in juvenile court where they face up to 10 years in prison and fines of up to $10,000 if convicted. Police officials explained that the investigation is still in progress, but the evidence so far overwhelmingly speaks to the guilt of the suspects."
arson_free = "Since mid-April 2018, a region in Bavaria, Germany, has faced a series of fires set by one or more vandals that ignited trash cans, a car, and even an entire town house. At its peak, 8 different fires were started in a single night. Fortunately, nobody was hurt, but the material damages are estimated at around $450,000. The police could not make significant advances in finding the perpetrator until they publicly offered a reward of $6000 for each helpful tip. A crumpled-up black shirt found at one of the crime scenes and bearing a unique design may have been the critical clue. About a week ago, the police arrested a 37 year old suspect on charges of second-degree arson. If found guilty, he will face up to 10 years in prison. According to recent reports though, experts don't believe that the existing evidence clearly points to him."
arson_jail = "Since mid-April 2018, a region in Bavaria, Germany, has faced a series of fires set by one or more vandals that ignited trash cans, a car, and even an entire town house. At its peak, 8 different fires were started in a single night. Fortunately, nobody was hurt, but the material damages are estimated at around $450,000. The police could not make significant advances in finding the perpetrator until they publicly offered a reward of $6000 for each helpful tip. A crumpled-up black shirt found at one of the crime scenes and bearing a unique design may have been the critical clue. About a week ago, the police arrested a 37 year old suspect on charges of second-degree arson. According to recent reports, experts believe that the existing evidence leaves no doubt about his guilt."
smuggler_free = "Thanks to a routine cargo vessel check, the Indonesian police were able to prevent a smuggling attempt. They found about 150 exotic birds stuffed into plastic drain pipes that were sealed at each end by a wire. The animals were meant to be transported to the US, where they were to be sold as pets. Officials assume that the operation was planned and executed by a wildlife trafficking ring. This delivery would have been worth approximately $250,000. Nearly one week after discovering the illegally captured birds, the police arrested four men who purportedly attempted to commit the smuggling. The suspects face up to five years in prison and a fine of up to $8000 if found guilty. The case rests on an anonymous tip, which provides a very weak case for the suspects' guilt. Moreover, all four suspects appear to have plausible alibis. Thus, it seems very unlikely that they were involved."
smuggler_jail = "Thanks to a routine cargo vessel check, the Indonesian police were able to prevent a smuggling attempt. They found about 150 exotic birds stuffed into plastic drain pipes that were sealed at each end by a wire. The animals were meant to be transported to the US, where they were to be sold as pets. Officials assume that the operation was planned and executed by a wildlife trafficking ring. This delivery would have been worth approximately $250,000. Nearly one week after discovering the illegally captured birds, the police arrested four men who purportedly attempted to commit the smuggling. The suspects face up to five years in prison and a fine of up to $8000 if found guilty. The case rests on several eyewitness reports, which provide a very strong case for the suspects' guilt. Moreover, the suspects' testimonies have been found to contain inconsistencies. Thus, it seems very likely that they were involved."
professor_free = "A professor in the Neuroscience department at Corsen University has been accused of sexual harassment. 11 women, all of them former graduate students in the department, have come forward with stories about Prof. Jim Smith. The list of allegations against Smith includes frequently commenting on women's appearance, forcing students to have meetings with him in his private home, and hosting several hot-tub parties that included nudity, which students were present at. Witnesses also claim that it was well-known that Smith engaged in a series of quid pro quo sexual relationships with his female graduate students. An independent investigation by an external law firm was commissioned. If found guilty of sexual harassment, Smith may be suspended for an indefinite amount of time or even fired, a rarity for tenured professors. The report by the official investigation is based on testimony about his character by over 100 of his current and former students, and indicates that the facts are much less clear than the original allegations suggest."
professor_jail = "A professor in the Neuroscience department at Corsen University has been accused of sexual harassment. 11 women, all of them former graduate students in the department, have come forward with stories about Prof. Jim Smith. The list of allegations against Smith includes frequently commenting on women's appearance, forcing students to have meetings with him in his private home, and hosting several hot-tub parties that included nudity, which students were present at. Witnesses also claim that it was well-known that Smith engaged in a series of quid pro quo sexual relationships with his female graduate students. An independent investigation by an external law firm was commissioned. If found guilty of sexual harassment, Smith may be suspended for an indefinite amount of time or even fired, a rarity for tenured professors. The report by the official investigation is based on testimony about his character by over 100 of his current and former students, and indicates that Smith clearly showed a pattern of predatory behavior."
scam_free = "On August 30 2018, 30,000 internet users in Spain received an email with the subject line 'Immediate response required - copyright infringement'. In these emails, the recipients were accused of illegally distributing pornographic content and were instructed to pay $11,500 by the end of September to avoid a lawsuit. Searching for the email address online leads to an official law firm website located in Barcelona. Shortly after the emails were sent out, the law firm disowned them, calling them a hoax. So far, police officials estimate that approximately 5% of the recipients have paid the money. This sums to more than $17 million that have been paid to the scammers. Now, nearly 4 weeks later, the police have tracked down the internet cafe from which the emails were sent. After observing the cafe's camera footage, the police arrested a man and a woman (both 52) who were allegedly responsible for the scam. If found guilty, they face up to 9 years in prison. However, news outlets recently acquired access to the camera footage and found that the video material is of very poor quality and therefore potentially unreliable."
scam_jail = "On August 30 2018, 30,000 internet users in Spain received an email with the subject line 'Immediate response required - copyright infringement'. In these emails, the recipients were accused of illegally distributing pornographic content and were instructed to pay $11,500 by the end of September to avoid a lawsuit. Searching for the email address online leads to an official law firm website located in Barcelona. Shortly after the emails were sent out, the law firm disowned them, calling them a hoax. So far, police officials estimate that approximately 5% of the recipients have paid the money. This sums to more than $17 million that have been paid to the scammers. Now, nearly 4 weeks later, the police have tracked down the internet cafe from which the emails were sent. After observing the cafe's camera footage, the police arrested a man and a woman (both 52) who were allegedly responsible for the scam. If found guilty, they face up to 9 years in prison. Moreover, news outlets recently acquired access to the camera footage and found that the video material is of very high quality and therefore undoubtedly reliable."

seeds = tibble(
  story_title = c("bees_free","bees_jail","arson_free","arson_jail","smuggler_free","smuggler_jail","professor_free","professor_jail","scam_free","scam_jail"), 
  reproduction = c(bees_free,bees_jail,arson_free,arson_jail,smuggler_free,smuggler_jail,professor_free,professor_jail,scam_free,scam_jail),
  generation = 0,
  chain = "0")

# collect reproductions from first study

# choose 5 full chains per story
selectChains <- function(condition){
  chains = df %>% 
    filter((generation == 5) & (story_title == condition)) %>% 
    select(chain) %>% 
    distinct() %>% 
    slice(1:5)
  return(chains)
}

target_chains = bind_rows(selectChains('arson_free'),selectChains('arson_jail'),selectChains('bees_free'),selectChains('bees_jail'),selectChains('professor_free'),selectChains('professor_jail'),selectChains('scam_free'),selectChains('scam_jail'),selectChains('smuggler_free'),selectChains('smuggler_jail'))

d_subj_ratings = df %>% 
  subset(chain %in% target_chains$chain) %>% 
  select(chain,generation,reproduction,story_text,story_title) %>%  
  # only choose rows with reproductions that don't occur as story_text in the next generation
  filter((reproduction %in% story_text | generation == 5)) %>% 
  group_by(chain,generation) %>% 
  sample_n(1) %>% 
  select(-story_text) %>% 
  bind_rows(seeds) 

# table(d_subj_ratings$chain,d_subj_ratings$generation)

# write.table(d_subj_ratings, file = here("experiments","02_main","02_subjective-ratings","trial_info","reproduction-data.csv"), sep=";", row.names = FALSE, qmethod = "double")

#######################
# Sub-Corpus analysis #
#######################

df_clean = d_subj_ratings %>%
  mutate(NumOfChars=str_length(reproduction)) %>% 
  mutate(NumOfWords=str_count(reproduction, boundary("word")))

# 0.9950305
cor(df_clean$NumOfChars,df_clean$NumOfWords)

plot_corpus_length = ggplot(data = df_clean, mapping = aes(x = generation, y = NumOfWords)) +
  # individual data points (jittered horizontally)
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
               fill = "#E6AB02",
               color = "black",
               size = 4) +
  ylab("Number of Words") +
  xlab("Generation")

ggsave(filename="corpus_length.png",plot=plot_corpus_length,path = here("writing","2019_cogsci","graphs"),width = 6,height = 4)

################################
# Subjective Rating Pilot Data #
################################

# randomly choose two complete chains that are not in main experiment (jail and free condition) for pilot
d_pilot = df %>% 
  filter(chain=="mMLRHJaK0CYrkHG1" | chain=="b2lFKlRgR8OxDUPI") %>% 
  select(chain,generation,reproduction,story_title)

write.table(d_pilot, file = here("experiments","01_pilot","09_subjective-ratings","trial_info","pilot-reproduction-data.csv"), sep=";", row.names = FALSE, qmethod = "double")
