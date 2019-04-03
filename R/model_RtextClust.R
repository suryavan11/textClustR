## This is to turn off scientific notation. default is 0 for keeping scientific notation
options(scipen=999) 

setwd('working directory')

library(readr) ### file read write functions
library(ggplot2)  ### plotting
library(ggrepel)
library(hues)
library(lubridate)  ### date
library(stringr) ### vectorized string conversions
library(tidyr) ### data manipulation
library(reshape2) ### data manipulation
library(magrittr) ### pipes
library(plyr) ### data manipulation, load before dplyr
library(RODBC) ### for connecting to ODBC connections previously set up in windows 64 bit administrator
library(sqldf)  ### for odbc connections
library(Rtsne)
library(tokenizers)
library(text2vec)
library(Matrix)
library(superheat)
library(cld2)
library(dplyr) ### piping and chaining operations. Load this package last as it is widely used and has some conflicts with other packages, especially plyr

source("R/utils.R" )  ## change path name based on your local path

############ read data ###############
stopwords_longlist = readLines('data/stopwords_longlist_abhi.txt')

######### chat text
data11 = read_delim('data/conv_Jan_2019_predictions.txt','|')
data1 = data11%>%rename(txt = msg_clean)%>%mutate(l2='na')
data1 = data1%>% sample_n(30000)

# ######### sample data from online
# data1 = read_tsv('data/r52-train-all-terms.txt', col_names = F )
# data1 = data1%>%rename(txt = X2, l2 = X1)%>% distinct(txt, .keep_all = TRUE)


### preprocessing ###########################

data2 = data1%>% distinct(txt, .keep_all = TRUE)
data2$language = cld2::detect_language(data2$txt, lang_code = FALSE)
data2 = data2%>%filter(language == 'ENGLISH')
data2$txt = preprocess.text.fn(data2$txt)

data3 = data2

### get.embeddings. may take a while ###############
embeddings.return.list = get.embeddings(data3, stopwords_longlist)  ### see control.param within get.embeddings function to adjust default parameters
word_vectors =  embeddings.return.list$word_vectors
doc_vectors =  embeddings.return.list$doc_vectors                          
dtm =  embeddings.return.list$dtm                          
vocab = embeddings.return.list$vocab
tokens = embeddings.return.list$tokens

### get a tsne matrix for plotting clustering results. This will be used later in the code. may take a while  #######                         
return.list = create.rtsne.matrix(doc_vectors, number.of.samples = 10000, seed = 80)
rtsne_out = return.list$rtsne_out
random.indices = return.list$random.indices


################# generate keywords (unsupervised) or provide keywords (seeded) ############

### get initial keywords for unsupervised clustering , may take a while
seed.words.orig = unsup.choose.keywords(data3, doc_vectors, vocab, nclust = 50, keywords.per.clust=3, plot=F)

##### keywords for seeded clustering 
# seed.words.orig = data.frame( 'words' = c('accessories','authorized','cancel_service','credit','exchange','power,button','port,add','reset','complaint,escalation','fraud','hotspot','outage','payment -arrangement', 'autopay', 'arrangement','insurance,crack','data,plan,roaming,netflix,unlimited','return','charge,fee','restore','unlock,sim','upgrade,jump,trade','website', 'troubleshoot,wifi,bluetooth,mode'),
#                               'reasons' = c('cr_accessories','cr_account_maintenance','cr_cancel_an_account_or_msisdn','cr_credits_and_adjustments','cr_device_exchange','cr_device_hardware_issues','cr_device_or_account_activation','cr_device_software_issues','cr_escalation_or_complaint','cr_fraud','cr_mobile_broadband','cr_network_or_coverage','cr_payment','cr_autopay','cr_payment_arrangement','cr_phone_protection_solutions','cr_rate_plan_or_plan_fit_analysis','cr_return_device','cr_review_bill','cr_suspend_or_restore_service','cr_unlock_device','cr_upgrade_device','cr_website_navigation', 'cr_how_to_use_device'),
#                               stringsAsFactors = FALSE)
# seed.words.orig$type = 'seeded'



########## read keywords from a file
# seed.words.orig = read_csv( 'keyword.summary20190312_modified for fraud.csv'  )%>%
#  select(newwords, reasons)%>%rename(words = newwords)



########### clustering algorithm ################ 

control.param = list(
  'num.iter' = 50,
  'max.keywords.per.topic' = 50,
  'alpha' = 0.9, ### alpha (between 0 to 1) controls how fast cluster centers move as new keywords are introduced. alpha 0 = fast movement, higher chance of losing the original meaning, alpha = 1 slower drift, original meaning is better retained. imagine mixing black and white colors. black = old keywords, white = new keywords. the mixture of the two colors is the new cluster center. alpha = 1 gives black, alpha = zero gives white, and alpha between 0 and 1 gives shades of grey
  'beta' = 0.1, ### beta (between 0 to 1) controls how much the cluster centers drift away from the starting point. imagine an elastic band tied between the starting cluster center and the new cluster center. beta close to 1 means the elastic band is very strong and the new center wont drift too much from the starting point. beta close to 0 means the elastic band is very weak and the new center can drift freely away from the starting center
  'cluster.merge.threshold' = 0.90, ### ranges from 0 to 1. closer to 1 means two clusters have to be very similar to be merged into one cluster. closer to 0 means clusters that are farther away from each other can also merge rapidly 
  
  'quality.threshold.to.add.keywords' = 0.95, ### if current cluster quality is > 0.95*max( cluster quality over all iterations) then add keyword one by one
  'quality.threshold.to.remove.keywords' = 0.5, ### if current cluster quality is < 0.55*max( cluster quality over all iterations) then remove keyword one by one (tries to reduce the influence of the cluster as it is poor quality). between add and remove thresholds, the current number of keywords is kept as-is
  
  'silhouette.threshold' = 0, ### clusters below this silhouette threshold are considered for elimination every  'remove.silh.every.niter' iterations
  'remove.silh.every.niter' = 4, ### more agressive removal will give better clusters, but may eliminate subtopics that have higher overlap with other clusters
  'silh.clusters.to.remove' = 1, ### how many clusters to remove every n iterations
  'fast.silhouette.calc' = F ### fast.silhouette.calc = T uses an approximate silhouette calculation (dissimilarity between docs and cluster centers). fast.silhouette.calc = F uses sampled inter-document distances, which can be slower for larger datasets 
  
)


return.list = cluster.reason.vectors(data3, dtm, word_vectors, doc_vectors, seed.words.orig, control.param)

data3 = return.list$fndf
reason.vectors = return.list$fnrv
orig.reason.vectors = return.list$fnorigrv
cluster.center.progress = return.list$cluster.center.progress
seed.words.iter = return.list$word.list
seed.words = return.list$seed.words


#### quality plots #####################################################
temp = seed.words.iter%>%
  group_by(reasons2,iteration)%>%
  summarize(quality = mean(quality, na.rm = T), silh = mean(silh,na.rm=T))%>%
  ungroup()%>%
  group_by(iteration)%>%
  summarize(quality = mean(quality,na.rm=T), silh = mean(silh, na.rm=T))%>%
  ungroup()

ggplot(temp, aes(iteration,quality)) + geom_line() + geom_point() 



############## cluster center progress plots ################################

temp = cluster.center.progress%>%
  mutate(reasons = str_replace_all(reasons2,'[0-9]+$','') )%>%
  rename(iteration = iter)

seed.words.iter1 = seed.words.iter%>%
  left_join(temp, by = c('iteration','reasons','reasons2'))


cluster.center.progress$reasons = str_replace_all(cluster.center.progress$reasons2,'[0-9]+$','')
temp1 = seed.words.iter1%>%
  group_by(reasons,reasons2)%>%
  filter(iteration==max(iteration))%>%
  arrange(metric, .by_group = TRUE)%>%
  summarize(newwords = str_c(unique(words), collapse = ','), iteration = max(iteration))%>%
  ungroup()

temp2 = seed.words%>%
  group_by(reasons,reasons2)%>%
  summarize(orig.words = str_c(unique(words), collapse = ','))%>%
  ungroup() 

temp3 = temp1%>%left_join(temp2, by = 'reasons2')

cluster.center.progress1 = cluster.center.progress%>%
  left_join(temp3, by = 'reasons2')%>%
  filter(iter<=iteration)

ggplot(cluster.center.progress1) + geom_line(aes(iter, improvement, col = as.factor(reasons2))) +
  theme(legend.position="none", axis.text.x = element_text(angle = 45, hjust = 1, size = 8))

###########   keywords summary  ##############################
keyword.summary = seed.words.iter1%>%
  group_by(reasons,reasons2)%>%
  filter(iteration== max(seed.words.iter1$iteration) & iteration != 1)%>%
  ungroup()%>%
  group_by(words)%>%
  mutate(repeatword.rn = row_number() , repeatword.counts = n() )%>%
  ungroup()%>% 
  group_by(reasons,reasons2)%>%
  arrange(desc(metric), .by_group = TRUE)%>%
  summarize(newwords = str_c(words, collapse = ','),
            wordcount = length(words),
            reasons3 = first(str_c(words, ifelse(repeatword.counts> 1,repeatword.rn, '') ) ),
            weights = str_c(round(metric,3) , collapse = ','),
            iteration = str_c(unique(iteration), collapse = ','))%>%
  ungroup() %>%
  left_join(seed.words%>%group_by(reasons2)%>%summarize(words = paste0(words,collapse=','))%>%ungroup(), by = 'reasons2')


################# plot silhouette ###################################
final.silh = calculate.silhouette(reason.vectors, doc_vectors, keyword.summary,
                                  plot.silhouette = T,  fast.silhouette.calc = F  )


# At this point, keyword.summary file can be written to disc as a csv and manually edited 
# be careful about manual edits! it can throw the results off
 


##################### predictions ###########################
predict.list = predict.clusters(keyword.summary, data3, word_vectors, doc_vectors, tokens)
data4 = predict.list$predictions
keyword.summary = predict.list$keyword.summary

  
temp =   data4%>%
  select(top1class)%>%
  mutate(top1class = str_remove_all(top1class,'^simcr_'))%>%
  left_join(keyword.summary%>%mutate(reasons2,reasons3)%>%
              mutate(top1class = str_remove_all(reasons2,'^cr_')), by = 'top1class' )%>%
  mutate(reasons = str_replace_all(reasons,'[0-9]+$|^cr_',''))%>%
  pull(reasons3)


g1 = plot.cluster.results(rtsne_out,  random.indices, temp)
g1




