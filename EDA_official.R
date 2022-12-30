################ Clearing everything ################

rm(list = ls())

################ Loading Packages ################

library(here)
library(tidyverse) 
library(ggplot2)
library(dplyr)
library(forcats)
library(forecast)
library(lubridate)
library(stringr)
library(dlookr)
library(corrplot)
library(rpart)
library(rpart.plot)
library(mlbench)
library(randomForest)
library(caret)
library(pROC)
library(gbm)
library(parallel)
library(ROCR)
library(pdp)
library(doParallel)
library(corrplot)
library(dygraphs)
library(xts)
library(glmnet)
library(magrittr)
library(ROSE)
library(sqldf)
library(scales)
library(ggrepel)

################ Reading dataset ################

dset <- read.csv(file = here("data/AT2_credit_train.csv"))

################ Summarizing Dataset ################

head(dset)
summary(dset)
str(dset)
colnames(dset)
sum(is.na(dset))

################ Cleaning dataset ################

## ID column dropped 

dset <- dset[-1] 

## Labeled sex column as a Gender

colnames(dset)[2] <- "Gender"

## Replacing invalid age

range(dset$AGE)
median(dset$AGE)
dset[c(which(dset$AGE>94)),5] = 34

## Grouping age by age groups
dset <- dset %>% 
  mutate(
    # Create categories
    age_group = dplyr::case_when(
      AGE < 18            ~ "0-17",
      AGE >= 18 & AGE <= 30 ~ "18-30",
      AGE > 31 & AGE <= 50 ~ "31-50",
      AGE > 50 & AGE <= 65 ~ "51-65",
      AGE > 65             ~ "> 65"
    ),
    # Convert to factor
    age_group = factor(
      age_group,
      level = c("0-17","18-30","31-50","51-65","> 65")
    )
  )


## Checking gender column 

unique(dset$Gender)

# Create another df just for EDA purpose

dset_eda <- dset

## Set Gender as factors 

dset$Gender <- as.factor(ifelse(dset$Gender == 1 , 1,
                                ifelse(dset$Gender == 2, 2, "NULL")))

dset_eda$Gender <- as.factor(ifelse(dset_eda$Gender == 1 , "Male" ,
                                    ifelse(dset_eda$Gender == 2, "Female", "NULL")))

## Checking gender column again
unique(dset$Gender)
unique(dset_eda$Gender)

## Set Education as factors
dset$EDUCATION <- as.factor(ifelse(dset$EDUCATION == 1 , 1 , 
                                   ifelse(dset$EDUCATION == 2 , 2 ,
                                          ifelse(dset$EDUCATION == 3 , 3 , 
                                                 ifelse(dset$EDUCATION == 4 , 4 , 5)))))

dset_eda$EDUCATION <- as.factor(ifelse(dset_eda$EDUCATION == 1 , "Graduate_School" , 
                                       ifelse(dset_eda$EDUCATION == 2 , "University" ,
                                              ifelse(dset_eda$EDUCATION == 3 , "High_school" , 
                                                     ifelse(dset_eda$EDUCATION == 4 , "Other" , "Unknown")))))   

## Set Marriage as factors
dset$MARRIAGE <- as.factor(dset$MARRIAGE)

dset_eda$MARRIAGE <- as.factor(ifelse(dset_eda$MARRIAGE == 1 , "Married" , 
                                      ifelse(dset_eda$MARRIAGE == 2 , "Single" , 
                                             ifelse(dset_eda$MARRIAGE == 3 , "Divorce" ,"Others" ))))




## Set default as factors 
dset$default <- (ifelse(dset$default == "Y" , 1 , 2)) 
dset_eda$default <- as.factor(ifelse(dset_eda$default == "Y" , "Yes" , "No")) 

################ Visualizing EDA ################
## Initial exploratory data analysis. Plotting charts of demographic breakouts

gender <- dset_eda %>%
  filter(Gender != 'NULL') %>% 
  group_by(Gender)%>%
  mutate(count_name_occurr = n())%>%
  ggplot(aes(x=reorder(Gender,count_name_occurr))) +  
  geom_bar(stat = "count", size = 0.5, fill = "#0070C0") +
  coord_flip() +
  geom_label(stat='count', aes(label=..count..), 
             hjust = 0.5, vjust = 0.5, size = 4.5, color = "grey3", 
             fill = "azure2") + 
  theme_classic() +
  theme(legend.position = "none")+
  labs(title = "Records per Gender",
       subtitle = " ",
       caption = " ",
       x = "Gender",
       y = "# of records")
gender


edu <- dset_eda %>%
  group_by(EDUCATION)%>%
  mutate(count_name_occurr = n())%>%
  ggplot(aes(x=reorder(EDUCATION,count_name_occurr))) + 
  geom_bar(stat = "count", fill = "#0070C0")+
  coord_flip()+
  geom_label(stat='count', aes(label=..count..), 
             hjust = 0.5, vjust = 0.5, size = 4.5, color = "grey3", 
             fill = "azure2") + 
  theme_classic() +
  theme(legend.position = "none")+
  labs(title = "Records per Education Level",
       subtitle = " ",
       caption = " ",
       x = "Education level",
       y = "# of records")

edu

marital <- dset_eda %>%
  group_by(MARRIAGE)%>%
  mutate(count_name_occurr = n())%>%
  ggplot(aes(x=reorder(MARRIAGE,count_name_occurr))) + 
  geom_bar(stat = "count", fill = "#0070C0")+
  coord_flip()+
  geom_label(stat='count', aes(label=..count..), 
             hjust = 0.5, vjust = 0.5, size = 4.5, color = "grey3", 
             fill = "azure2") + 
  theme_classic() +
  theme(legend.position = "none")+
  labs(title = "Records per Marriage status",
       subtitle = " ",
       caption = " ",
       x = "Marriage status",
       y = "# of records")
marital

## Set payment columns as factors
dset[6:11] <- lapply(dset[6:11], factor) 
dset_eda[6:11] <- lapply(dset_eda[6:11], factor) 

p0 <- ggplot(dset_eda, aes(PAY_0)) + 
  geom_bar(fill = "#0070C0")+
  geom_label(stat='count', aes(label=..count..), 
             hjust = 0.5, vjust = 0.5, size = 4.5, color = "grey3", 
             fill = "azure2")+
  theme_classic() +
  theme(legend.position = "none")+
  labs(title = "PAY_0 distribution",
       subtitle = " ",
       caption = " ",
       x = "PAY_0",
       y = "# of records")


p0

p2 <- ggplot(dset_eda, aes(PAY_2)) + 
  geom_bar(fill = "#0070C0")+
  geom_label(stat='count', aes(label=..count..), 
             hjust = 0.5, vjust = 0.5, size = 4.5, color = "grey3", 
             fill = "azure2")+
  theme_classic() +
  theme(legend.position = "none")+
  labs(title = "PAY_2 distribution",
       subtitle = " ",
       caption = " ",
       x = "PAY_2",
       y = "# of records")
p2

p3 <- ggplot(dset_eda, aes(PAY_3)) + 
  geom_bar(fill = "#0070C0")+
  geom_label(stat='count', aes(label=..count..), 
             hjust = 0.5, vjust = 0.5, size = 4.5, color = "grey3", 
             fill = "azure2")+
  theme_classic() +
  theme(legend.position = "none")+
  labs(title = "PAY_3 distribution",
       subtitle = " ",
       caption = " ",
       x = "PAY_3",
       y = "# of records")
p3

p4 <- ggplot(dset_eda, aes(PAY_4)) + 
  geom_bar(fill = "#0070C0")+
  geom_label(stat='count', aes(label=..count..), 
             hjust = 0.5, vjust = 0.5, size = 4.5, color = "grey3", 
             fill = "azure2")+
  theme_classic() +
  theme(legend.position = "none")+
  labs(title = "PAY_4 distribution",
       subtitle = " ",
       caption = " ",
       x = "PAY_4",
       y = "# of records")
p4

p5 <- ggplot(dset_eda, aes(PAY_5)) + 
  geom_bar(fill = "#0070C0")+
  geom_label(stat='count', aes(label=..count..), 
             hjust = 0.5, vjust = 0.5, size = 4.5, color = "grey3", 
             fill = "azure2")+
  theme_classic() +
  theme(legend.position = "none")+
  labs(title = "PAY_5 distribution",
       subtitle = " ",
       caption = " ",
       x = "PAY_5",
       y = "# of records")
p5

p6 <- ggplot(dset_eda, aes(PAY_6)) + 
  geom_bar(fill = "#0070C0")+
  geom_label(stat='count', aes(label=..count..), 
             hjust = 0.5, vjust = 0.5, size = 4.5, color = "grey3", 
             fill = "azure2")+
  theme_classic() +
  theme(legend.position = "none")+
  labs(title = "PAY_6 distribution",
       subtitle = " ",
       caption = " ",
       x = "PAY_6",
       y = "# of records")
p6


## Visualize credit limits. We can identify 1,000,000 as an outlier
limit <- ggplot(dset_eda) + 
  geom_qq(aes(sample=LIMIT_BAL))+
  theme_classic() +
  theme(legend.position = "none")+
  labs(title = "Credit Limits",
       subtitle = " ",
       caption = " ")
limit

## Visualize credit limits compared to default. Overall, people who default have lower credit limits.
limit2 <- ggplot(dset_eda, aes(default, LIMIT_BAL)) + 
  geom_boxplot(fill="#0070C0")+
  theme_classic() +
  theme(legend.position = "none")+
  labs(title = "Credit Limits vs Default",
       subtitle = " ",
       caption = " ",
       x = "Default",
       y = "Credit limit")
limit2

## Visualize credit limits compared to education

limit3 <- ggplot(dset_eda, aes(EDUCATION, LIMIT_BAL, fill = EDUCATION)) + 
  geom_boxplot(fill="#0070C0")+
  theme_classic() +
  theme(legend.position = "none")+
  labs(title = "Credit Limits vs Education Level",
       subtitle = " ",
       caption = " ",
       x = "Education level",
       y = "Credit limit")
limit3

## List all limit balance values

unique(dset_eda$LIMIT_BAL)

## Proportion of cards that default
prop.table(table(dset_eda$default))

## Credit default by Gender
dset_eda %>% 
  dplyr::filter(Gender != "NULL") %>% 
  ggplot(aes(x = Gender, fill = default)) +
  geom_bar() +
  labs(x = 'Gender',title="Credit Default by Gender") +
  theme_classic() +
  stat_count(aes(label = ..count..), geom = "label")

## Credit default by Education
dset_eda %>% 
  ggplot(aes(x = EDUCATION, fill = default)) +
  geom_bar(position = 'fill') +
  labs(x = 'Education Level',title="Credit Default by Education") +
  theme_classic()

## Credit default by Marital status
dset_eda %>% 
  ggplot(aes(x = MARRIAGE, fill = default)) +
  geom_bar(position = 'fill') +
  labs(x = 'Marital status',title="Credit Default by Marital status") +
  theme_classic()

## Credit default by Payment status
dset_eda %>% 
  ggplot(aes(x = PAY_0, fill = default)) +
  geom_bar(position = 'fill') +
  labs(x = 'Payment Status',title="Payment status in sept v/s Default rate") +
  theme_classic()

#create new variable
# female
female_def_percent <- dset_eda %>% 
  filter(Gender=="Female") %>% 
  group_by(default) %>% 
  tally()

# male
male_def_percent <- dset_eda %>% 
  filter(Gender=="Male") %>% 
  group_by(default) %>% 
  tally()

# calculate percentage
female_def_percent %>%
  arrange(desc(n)) %>%
  mutate(prop = percent(n / sum(n))) -> female_def_percent

male_def_percent %>%
  arrange(desc(n)) %>%
  mutate(prop = percent(n / sum(n))) -> male_def_percent

# Default percentage by gender. In general, male tends to default more comparing to female
# Female default - pie chart
ggplot(female_def_percent, aes(x = "", y = n, fill = fct_inorder(default))) + 
  geom_bar(stat = "identity", width = 1) +
  geom_col(color = "black", width = 1) +
  coord_polar("y", start = 0) + 
  geom_label_repel(aes(label = prop), size=5, show.legend = F, nudge_x =1, nudge_y = 1) +
  labs(
    title = "Female Default Percentage"
  ) +
  scale_fill_brewer(palette = "Blues") +
  theme_classic() +
  guides(fill = guide_legend(title = "Default")) +
  ggthemes::theme_tufte() +
  theme(plot.title = element_text(size = 15L, hjust = 0.5))

# Male default - pie chart
ggplot(male_def_percent, aes(x = "", y = n, fill = fct_inorder(default))) + 
  geom_bar(stat = "identity", width = 1) +
  geom_col(color = "black", width = 1) +
  coord_polar("y", start = 0) + 
  geom_label_repel(aes(label = prop), size=5, show.legend = F, nudge_x =1, nudge_y = 1) +
  labs(
    title = "Male Default Percentage"
  ) +
  scale_fill_brewer(palette = "Blues") +
  theme_classic() +
  guides(fill = guide_legend(title = "Default")) +
  ggthemes::theme_tufte() +
  theme(plot.title = element_text(size = 15L, hjust = 0.5))

## Credit default by Age Group
dset_eda %>%
  filter(!is.na(age_group)) %>%
  ggplot() +
  aes(x = age_group, fill = default) +
  geom_bar() +
  scale_fill_hue(direction = 1) +
  theme_minimal()+
  labs(title = "Credit Default by age group",
       subtitle = " ",
       caption = " ",
       x = "Age group",
       y = "# of records")

