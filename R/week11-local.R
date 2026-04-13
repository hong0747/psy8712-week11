# Script Settings and Resources
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
set.seed(394587)
library(tidyverse)
library(caret)
library(haven)
library(jtools)
library(parallel)
library(doParallel)

# Data Import and Cleaning
gss_import_tbl <- read_spss(file = "../data/GSS2016.sav", user_na = TRUE) %>% 
  filter(!is.na(mosthrs))
gss_tbl <- gss_import_tbl %>%
  select(-hrs1, -hrs2) %>%
  select(where(~ mean(is.na(.)) < .75)) %>%
  mutate(across(everything(), as.numeric))

# Visualization
ggplot(gss_tbl, 
       aes(x = mosthrs)) +
  geom_histogram()

# Analysis
holdout_indices <- createDataPartition(gss_tbl$mosthrs, 
                                       p = .25, 
                                       list=F)
gss_holdout <- gss_tbl[holdout_indices,]
gss_training <- gss_tbl[-holdout_indices,]

model1_et <- system.time({
  model1 <- train(
    mosthrs ~ .,
    gss_training,
    method = "lm",
    na.action = na.pass,
    preProcess=c("medianImpute","center","nzv","scale"),
    trControl=trainControl(
      method="cv", number=10, verboseIter=T
    )
  )
})
model1_et

hocv_cor_1 <- cor(
  predict(model1, gss_holdout, na.action=na.pass),
  gss_holdout$mosthrs
)^2

model2_et <- system.time({
  model2 <- train(
    mosthrs ~ .,
    gss_training,
    method = "glmnet",
    na.action = na.pass,
    preProcess=c("medianImpute","center","nzv","scale"),
    trControl=trainControl(
      method="cv", number=10, verboseIter=T
    )
  )
})
model2_et

hocv_cor_2 <- cor(
  predict(model2, gss_holdout, na.action=na.pass),
  gss_holdout$mosthrs
)^2

model3_et <- system.time({
  model3 <- train(
    mosthrs ~ .,
    gss_training,
    method = "ranger",
    na.action = na.pass,
    tuneLength = 1,
    preProcess=c("medianImpute","center","nzv","scale"),
    trControl=trainControl(
      method="cv", number=10, verboseIter=T
    )
  )
})
model3_et

hocv_cor_3 <- cor(
  predict(model3, gss_holdout, na.action=na.pass),
  gss_holdout$mosthrs
)^2

model4_et <- system.time({
  model4 <- train(
    mosthrs ~ .,
    gss_training,
    method = "xgbLinear",
    na.action = na.pass,
    tuneLength = 1,
    preProcess=c("medianImpute","center","nzv","scale"),
    trControl=trainControl(
      method="cv", number=10, verboseIter=T
    )
  )
})
model4_et

hocv_cor_4 <- cor(
  predict(model4, gss_holdout, na.action=na.pass),
  gss_holdout$mosthrs
)^2

summary(resamples(list("lm"=model1, "glmnet"=model2, "ranger"=model3, "xgbLinear"=model4)))
dotplot(resamples(list("lm"=model1, "glmnet"=model2, "ranger"=model3, "xgbLinear"=model4)))

num_cores <- detectCores()
local_cluster <- makeCluster(floor(num_cores * 0.75))
registerDoParallel(local_cluster)

model1_pl_et <- system.time({
  model1_pl <- train(
    mosthrs ~ .,
    gss_training,
    method = "lm",
    na.action = na.pass,
    preProcess=c("medianImpute","center","nzv","scale"),
    trControl=trainControl(
      method="cv", number=10, verboseIter=T
    )
  )
})
model1_pl_et

hocv_cor_1_pl <- cor(
  predict(model1_pl, gss_holdout, na.action=na.pass),
  gss_holdout$mosthrs
)^2

model2_pl_et <- system.time({
  model2_pl <- train(
    mosthrs ~ .,
    gss_training,
    method = "glmnet",
    na.action = na.pass,
    preProcess=c("medianImpute","center","nzv","scale"),
    trControl=trainControl(
      method="cv", number=10, verboseIter=T
    )
  )
})
model2_pl_et

hocv_cor_2_pl <- cor(
  predict(model2_pl, gss_holdout, na.action=na.pass),
  gss_holdout$mosthrs
)^2

model3_pl_et <- system.time({
  model3_pl <- train(
    mosthrs ~ .,
    gss_training,
    method = "ranger",
    na.action = na.pass,
    tuneLength = 1,
    preProcess=c("medianImpute","center","nzv","scale"),
    trControl=trainControl(
      method="cv", number=10, verboseIter=T
    )
  )
})
model3_pl_et

hocv_cor_3_pl <- cor(
  predict(model3_pl, gss_holdout, na.action=na.pass),
  gss_holdout$mosthrs
)^2

model4_pl_et <- system.time({
  model4_pl <- train(
    mosthrs ~ .,
    gss_training,
    method = "xgbLinear",
    na.action = na.pass,
    tuneLength = 1,
    preProcess=c("medianImpute","center","nzv","scale"),
    trControl=trainControl(
      method="cv", number=10, verboseIter=T
    )
  )
})
model4_pl_et

hocv_cor_4_pl <- cor(
  predict(model4_pl, gss_holdout, na.action=na.pass),
  gss_holdout$mosthrs
)^2

# Publication
table1_tbl <- tibble(
  algo = c("lm", "glmnet", "ranger", "xgbLinear"),
  cv_rsq = c(
    str_remove(round(max(model1$results$Rsquared),2),"^0"),
    str_remove(round(max(model2$results$Rsquared),2),"^0"),
    str_remove(round(max(model3$results$Rsquared),2),"^0"),
    str_remove(round(max(model4$results$Rsquared),2),"^0")
  ),
  ho_rsq = c(
    str_remove(round(hocv_cor_1,2),"^0"),
    str_remove(round(hocv_cor_2,2),"^0"),
    str_remove(round(hocv_cor_3,2),"^0"),
    str_remove(round(hocv_cor_4,2),"^0")
  )
)

table2_tbl <- tibble(
  original = c("lm" = as.numeric(model1_et[3]), "glmnet" = as.numeric(model2_et[3]), "ranger" = as.numeric(model3_et[3]), "xgbLinear" = as.numeric(model4_et[3])),
  parallel = c("lm" = as.numeric(model1_pl_et[3]), "glmnet" = as.numeric(model2_pl_et[3]), "ranger" = as.numeric(model3_pl_et[3]), "xgbLinear" = as.numeric(model4_pl_et[3]))
)

write_csv(x = table1_tbl, "../out/table1.csv")
write_csv(x = table2_tbl, "../out/table2.csv")

# Which models benefited most from parallelization and why?
# The random forest and extreme gradient boosting models benefited the most from parellelization, likely because they are higher in complexity.
  
# How big was the difference between the fastest and slowest parallelized model? Why?
# For the OLS model, there was a ~4% increase in time. For the random forest model, there was a ~67% decrease in time. This is likely because the OLS model is simpler than the random forest model.
   
# If your supervisor asked you to pick a model for use in a production model, which would you recommend and why? Consider both Table 1 and Table 2 when providing an answer.
# I would recommend either the random forest or the extreme gradient boosting models because they have larger holdout R squared values and their execution times are not much slower than the other models when parallelized.