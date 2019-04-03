
preprocess.text.fn <- function(x) {
  x = str_replace_all( str_to_lower(x),'[^a-z ]','' )
  x = str_replace_all( str_trim(x),'\\s+',' ' )
  return(x)
}

tokenize.fn <- function(x, stopwords.list) {
  tokens = x%>%tokenize_word_stems(stopwords =  stopwords.list ) ## stopwords::stopwords('en')) 
  tokens = lapply(tokens,function(x) x[nchar(x)>2])
  tokens = lapply(tokens,function(x) x[nchar(x)<= 20])
  tokens = lapply(tokens,function(x) x[!x %in% stopwords.list])
  return(tokens)
}

get.embeddings <- function(fndf, stopword.list, control.param = list()  ) {
  
  if (length(control.param) == 0 ) {
    
    control.param = list(
      
      'vocab.ngram_min' = 1,
      'vocab.ngram_max' = 2,
      
      'prune.vocab.term.count.min' = 10,
      'prune.vocab.doc.count.min' = 10,
      'prune.vocab.doc.prop.max' = 0.5,
      
      'tfidf.norm' = 'none',
      
      'glove.skip.grams.window' = 100,
      'glove.word.vectors.size' = 50,
      'glove.x.max' = 10,
      'glove.n.iter' = 50, 
      'glove.convergence.tol' = 0.01
      
    ) 
  }
  
  
  ### tokenize
  tokens = tokenize.fn(fndf$txt,stopword.list)
  it = itoken(tokens, progressbar = FALSE)
  vocab <- create_vocabulary(it, 
                             ngram = c(ngram_min = control.param$vocab.ngram_min,
                                       ngram_max = control.param$vocab.ngram_max),
                             sep_ngram = "_" )
  vocab <- prune_vocabulary(vocab, term_count_min = control.param$prune.vocab.term.count.min,
                            doc_count_min = control.param$prune.vocab.doc.count.min, 
                            doc_proportion_max = control.param$prune.vocab.doc.prop.max ) ### play around with these parameters
  vectorizer <- vocab_vectorizer(vocab)
  model_tfidf = TfIdf$new(norm= control.param$tfidf.norm )
  dtm = create_dtm(it, vectorizer) %>% 
    fit_transform(model_tfidf)  
  
  
  ########### train a glove model to find similar keywords, might take some time to run
  tcm <- create_tcm(it, vectorizer, skip_grams_window = control.param$glove.skip.grams.window ) ## 5,30, 100
  glove = GlobalVectors$new(word_vectors_size = control.param$glove.word.vectors.size,
                            vocabulary = vocab, 
                            x_max = control.param$glove.x.max)
  wv_main = glove$fit_transform(tcm, 
                                n_iter = control.param$glove.n.iter, 
                                convergence_tol = control.param$glove.convergence.tol)
  wv_context = glove$components
  word_vectors = wv_main # + t(wv_context)
  
  
  
  
  ###### create document vectors by averaging word vectors
  common_terms = intersect(colnames(dtm), rownames(word_vectors) )
  dtm_averaged = dtm[, common_terms]
  # dtm_averaged =  normalize(dtm[, common_terms], "l1")
  # you can re-weight dtm above with tf-idf instead of "l1" norm
  doc_vectors = dtm_averaged %*% word_vectors[common_terms, ]
  
  
  return.list = list(
    'tokens' = tokens,
    'vocab' = vocab,
    'vectorizer' = vectorizer,
    'word_vectors' = word_vectors,
    'doc_vectors' = doc_vectors,
    'dtm' = dtm
    
  )
  
  return(return.list)
  
}

get.metric = function(fndtm,fnrv,fnwv, fndv, cosine.threshold = 0.8, use.dist = F) {
  
  fnrv = fnrv%>%
    select(starts_with('X'))%>%
    as.matrix()
  
  rw = sim2(fnrv, fnwv, method = 'cosine', norm = 'l2')  ### cosine-reason & word
  dw = fndtm[,colnames(rw), drop = F]   ### doc word matrix
  dr =  sim2(fndv, fnrv, method = 'cosine', norm = 'l2') 
  
  rw = rw[colnames(dr),colnames(dw), drop = F]   ### ensure that all matrices are lined up correctly before any matrix operation
  
  
  if (use.dist == T) {
    ### dist matrix
    
    indices =   apply(dr, 2, function(x) sample(which(x>0), min(length(which(x>0)),5000, na.rm = T)  ) )  
    dr.mod = dr
    dr.mod[,] = NA
    
    for (ind in 1:dim(dr)[2]) { 
      # print(ind)
      dd =   sim2(fndv[indices[[ind]],] ,  method = 'cosine', norm = 'l2') 
      dd.soft = 1/(2-dd) 
      dr.mod[indices[[ind]],ind] = (dd.soft %*% dr[indices[[ind]],1,drop = F] )/rowSums(dd.soft)
    }
    
  }
  # metric1 = as.matrix( (t(dw) %*% dr)/(t(dw) %*% (dr*0 + 1) )  )
  
  ### wd x dr / (denom) is the weighted average of dr similarities by the tfidf counts (the denom is essentially just a sum of the tfidf weights per word. the *0+1 part creates a matrix of ones)
  ### the above weighted average says whether a word is tied to a reason, but does not account for its ties to other reasons. multiplying by t(rw) brings that in

  if (use.dist == T) {
    metric = as.matrix( (t(dw) %*% dr.mod)/(t(dw) %*% (dr*0 + 1) ) * t(rw) )
  } else {
    metric = as.matrix( (t(dw) %*% dr)/(t(dw) %*% (dr*0 + 1) ) * t(rw) )
  }
  
 
  ##################################################################
  
  
  word.list = as.data.frame(metric)%>%
    mutate(words = rownames(metric))%>%
    melt(id.vars = 'words', variable.name = 'reasons2', value.name = 'metric') %>%
    mutate(reasons = str_replace_all(reasons2,'[0-9]+$',''))%>%
    filter(metric>0) # %>%
  # group_by(reasons2)%>%
  # mutate(metric = exp(-1*row_number(desc(metric)))*metric )%>%
  # ungroup()
  
  

  
  return(word.list)
  
  
}

calculate.silhouette <- function(fnrv, fndv, fnkey.sum = NULL, plot.silhouette = T, fast.silhouette.calc = F) {
  
  if (!is.null(fnkey.sum)) {
    rownames(fnrv) = as.character ( data.frame('reasons2' = rownames(fnrv))%>%
                                      left_join(fnkey.sum, by = 'reasons2')%>%
                                      pull(reasons3) )
    
  }
  
  
  fnrv = fnrv%>%
    select(starts_with('X'))%>%
    as.matrix()
  
  dr =  sim2(fndv, fnrv, method = 'cosine', norm = 'l2') 
  
  
  membership =   t(apply(dr, 1, function(x) as.numeric(x != max(x))))
  membership[membership == 0] = NA
  membership1 = colnames(dr)[apply(dr,1, which.max)]
  ## dr.dissim = acos(round(dr, 10)) / pi
  dr.dissim = round(1-dr,3)
  
  
  if (fast.silhouette.calc == T) {
    ### https://rlbarter.github.io/superheat-examples/Word2Vec/
    ### this is not the true silhouette calculation. the true silhouette calc would need a distance matrix between docs
    silh  = apply(dr.dissim * membership,1,min) - apply(dr.dissim,1,min)
    
  } else {
    
    #### alternative silh calculation
    indices =   apply(membership, 2, function(x) sample(which(is.na(x)), min(length(which(is.na(x))),5000, na.rm = T)  ) )  
    ai = as.numeric(rep(NA,nrow(dr)) )
    
    
    for (ind in 1:dim(dr)[2]) { 
      # print(ind)
      if (length(indices[[ind]]) != 0  ) {
        dd.dissim =  1- sim2(fndv[indices[[ind]],,drop=F] ,  method = 'cosine', norm = 'l2') 
        ai[  indices[[ind]]  ] = rowMeans(dd.dissim)
      }
    }
    
    bi = apply(dr.dissim * membership,1,function(x) min(x, na.rm = T)  )
    bi[bi==Inf] = NA   ### Inf are coming from short texts that are not assigned to any cluster
    
    silh =  bi - ai
    
  }
  
  #### quality/coherence calculation
  temp =  t(apply(dr, 1, function(x) as.numeric(x == max(x))))
  temp[temp==0] = NA
  qual = colSums(dr*temp, na.rm = T)
  
  
  temp = as.matrix(dr )
  temp.silh = data.frame(membership1, silh )%>%group_by(membership1)%>%summarize(silh = mean(silh))%>%ungroup()
  temp.silh = rbind(temp.silh,
                    data.frame('membership1' = colnames(temp)[!colnames(temp) %in% unique(temp.silh$membership1)],
                               'silh' = rep(0,length(colnames(temp)[!colnames(temp) %in% unique(temp.silh$membership1)] )) 
                    ) )
  
  temp.silh = temp.silh%>%left_join(data.frame('quality' = qual, 'membership1' = names(qual)), by = 'membership1' )
  
  if (plot.silhouette == T) {
    g1 = superheat(temp, 
                   
                   # row and column clustering
                   membership.rows = membership1 ,
                   membership.cols = colnames(temp),
                   
                   # top plot: silhouette
                   yt = temp.silh$silh[match(colnames(temp) , temp.silh$membership1 )],  ### this is a little tricky. the order of yt should match the original colnames(temp) order, not the order(colnames(temp)) order
                   yt.axis.name = "Cosine\nsilhouette\nwidth",
                   yt.plot.type = "bar",
                   yt.bar.col = "grey35",
                   
                   # order of rows and columns within clusters
                   order.rows = order(membership1 ),
                   order.cols = order(colnames(temp)),
                   
                   # bottom labels
                   bottom.label.col = c("grey95", "grey80"),
                   bottom.label.text.angle = 90,
                   bottom.label.text.alignment = "right",
                   bottom.label.size = 0.28,
                   
                   # left labels
                   left.label.col = c("grey95", "grey80"),
                   left.label.text.alignment = "right",
                   #left.label.size = 0.26,
                   
                   # smooth heatmap within clusters
                   smooth.heat = T,
                   
                   # title
                   title = "(b)")
    
    g1
    
  }
  
  return(temp.silh)
  
}

get.reason.vectors <- function(fnoldrv,fnorigrv, fnwv,word.list, weigh.by.similarity = T, alpha = 0, beta = 0.8 ) {
  
  
  
  fnrv =data.frame(fnwv[rownames(fnwv) %in% word.list$words,,drop = FALSE])%>%
    mutate(words = row.names(.))%>%
    left_join(word.list, by = 'words')
  
  fnrv[,str_detect(colnames(fnrv),'^X')] = fnrv[,str_detect(colnames(fnrv),'^X')] * fnrv$sign
  
  fnrv = fnrv%>%
    select(reasons,reasons2,metric,words,everything())%>%select(-words, -sign)
  
  ### weighting by simlarity. w1*x1/sum(w) is calculated for each line and then sum  at the end (this rowwise calculation is weighted average and replaces mean)
  if (weigh.by.similarity == T) {
    fnrv[,str_detect(colnames(fnrv),'^X')] = fnrv[,str_detect(colnames(fnrv),'^X')]*fnrv$metric
    fnrv = fnrv%>%
      group_by(reasons,reasons2)%>%
      mutate(metric = 1/sum(metric))%>%
      ungroup()
    fnrv[,str_detect(colnames(fnrv),'^X')] = fnrv[,str_detect(colnames(fnrv),'^X')]*fnrv$metric
    fnrv = fnrv%>%select(-metric)%>%group_by(reasons,reasons2)%>%summarize_all(funs(sum))%>%ungroup()
    rownames(fnrv) = fnrv$reasons2
    
  } else {
    
    fnrv = fnrv%>%select(-metric)%>%group_by(reasons,reasons2)%>%summarize_all(funs(mean))%>%ungroup()
    rownames(fnrv) = fnrv$reasons2
    
  }
  
  if(is.null(fnoldrv)) {
    return(fnrv)
  } else { 
    
    
    fnrv = fnrv%>%
      select(starts_with('X'))%>%
      as.matrix()
    
    fnoldrv = fnoldrv%>%
      select(starts_with('X'))%>%
      as.matrix()
    fnoldrv = fnoldrv[rownames(fnrv),]
    
    fnorigrv = fnorigrv%>%
      select(starts_with('X'))%>%
      as.matrix()
    fnorigrv = fnorigrv[rownames(fnrv),]
    
    
    temp = psim2(fnrv, fnorigrv,method = 'cosine', norm = 'l2') - beta
    temp[temp<0] = 0
    temp = 1-temp
    temp[temp<alpha] = alpha
    
    fnrv = fnrv * (1-temp) + fnoldrv * temp
    fnrv = data.frame(fnrv)%>%
      mutate(reasons2 = rownames(.), reasons = str_remove_all(reasons2,'[0-9]+$') )
    rownames(fnrv) = fnrv$reasons2
    
    return(fnrv)
  }
  
  
}

unsup.choose.keywords <- function(fndf, fndv, fnvocab, nclust = 50,keywords.per.clust=3, plot = F) {
  ################### choose seed keywords by hclust ########################
  set.seed(80)
  random.indices = sample(seq(1,nrow(fndv),1), min(nrow(fndv),10000), replace = F )
  doc.vectors.subset = as.matrix(fndv[random.indices,]) 
  distMatrix <- dist(doc.vectors.subset, method="euclidean")
  
  clust <- hclust(distMatrix,method="ward.D")
   plot(clust, cex=0.9, hang=-1)
  rect.hclust(clust, k=nclust)
  groups<-cutree(clust, k=nclust)
  
  if (plot==T) {
    rtsne_out <- Rtsne(doc.vectors.subset , perplexity = 50 ,check_duplicates = FALSE)
    
    tsne_plot <- data.frame(x = rtsne_out$Y[,1],
                            y = rtsne_out$Y[,2],
                            desc = groups
    )
    
    
    
    tsne_labels = tsne_plot%>%group_by(desc)%>%summarize(x = median(x), y = median(y))%>%ungroup()
    
    
    
    ggplot(tsne_plot%>%sample_n(5000), aes(x = x, y = y)) +
      stat_density2d(aes(fill = ..density.., alpha = 1), geom = 'tile', contour = F) +
      scale_fill_distiller(palette = 'Greys') + ###RdYlBu
      geom_point(aes(color=as.factor(as.integer(desc)) ),  size= 3, alpha = 1 ) +
      ggrepel::geom_label_repel(data = tsne_labels,
                                aes(x=x, y=y,label=desc, color = as.factor(as.integer(desc))),
                                fontface = 'bold' ) +
      scale_color_manual(values=rev(hues::iwanthue(200))) +
      theme(legend.position="none")
    
  }
  
  
  data.temp = fndf[random.indices,]
  
  
  for (i in seq_along(unique(groups) )) {
    print(i)
    df = data.temp%>%
      mutate(flag = ifelse( groups==i,2,1) )%>%
      group_by(flag) %>%
      summarize(txt = str_c(txt, collapse = ' ', sep = ' '))%>%
      ungroup()
    
    tokens.temp = tokenize.fn(df$txt,stopwords_longlist)
    it.temp = itoken(tokens.temp, progressbar = FALSE)
    vocab.temp <- create_vocabulary(it.temp, ngram = c(ngram_min = 1L, ngram_max = 2L), sep_ngram = "_" )
    vocab.temp <- prune_vocabulary(vocab.temp, term_count_min = 50L)
    vocab.temp = vocab.temp[vocab.temp$term %in% fnvocab$term,]
    vectorizer.temp <- vocab_vectorizer(vocab.temp)
    dtm.temp = create_dtm(it.temp, vectorizer.temp) 
    
    
    
    if(i==1) {
      seed.words.orig.temp = as.data.frame(t(as.matrix(dtm.temp)))%>%
        mutate(ratios = (`2`+1)/(`1`+1), words = rownames(.), 
               reasons = str_c('cr_', first(words[order(ratios,decreasing = T)]) ) )  %>%
        top_n(keywords.per.clust,ratios)%>% slice(seq_len(keywords.per.clust))  
      
    } else {
      seed.words.orig.temp = rbind(seed.words.orig.temp, as.data.frame(t(as.matrix(dtm.temp)))%>%
                                     mutate(ratios = (`2`+1)/(`1`+1), words = rownames(.),
                                            reasons =  str_c('cr_', first(words[order(ratios,decreasing = T)]) ) )  %>%
                                     top_n(keywords.per.clust,ratios)%>%slice(seq_len(keywords.per.clust))
      )
    }
    
  }
  
  
  seed.words.orig = seed.words.orig.temp %>%
    arrange(desc(ratios)) %>%
    select(words,reasons)%>%
    mutate(type = 'unsupervised')
  
  return(seed.words.orig)
  
}

cluster.reason.vectors <- function(fndf, fndtm, fnwv,fndv, orig.word.list, control.param = list() ) {
  
  if (length(control.param) == 0 ) {
    control.param = list(
      'num.iter' = 50,
      'max.keywords.per.topic' = 50,
      'alpha' = 0.9, ### alpha (between 0 to 1) controls how fast cluster centers move as new keywords are introduced. alpha 0 = fast movement, higher chance of losing the original meaning, alpha = 1 slower drift, original meaning is better retained. imagine mixing black and white colors. black = old keywords, white = new keywords. the mixture of the two colors is the new cluster center. alpha = 1 gives black, alpha = zero gives white, and alpha between 0 and 1 gives shades of grey
      'beta' = 0.1, ### beta (between 0 to 1) controls how much the cluster centers drift away from the starting point. imagine an elastic band tied between the starting cluster center and the new cluster center. beta close to 1 means the elastic band is very strong and the new center wont drift too much from the starting point. beta close to 0 means the elastic band is very weak and the new center can drift freely away from the starting center
      'cluster.merge.threshold' = 0.9, ### ranges from 0 to 1. closer to 1 means two clusters have to be very similar to be merged into one cluster. closer to 0 means clusters that are farther away from each other can also merge rapidly 
      
      'quality.threshold.to.add.keywords' = 0.95, ### if current cluster quality is > 0.95*max( cluster quality over all iterations) then add keyword one by one
      'quality.threshold.to.remove.keywords' = 0.5, ### if current cluster quality is < 0.55*max( cluster quality over all iterations) then remove keyword one by one (tries to reduce the influence of the cluster as it is poor quality). between add and remove thresholds, the current number of keywords is kept as-is
      
      'silhouette.threshold' = 0, ### clusters below this silhouette threshold are considered for elimination every  'remove.silh.every.niter' iterations
      'remove.silh.every.niter' = 4, ### more agressive removal will give better clusters, but may eliminate subtopics that have higher overlap with other clusters
      'silh.clusters.to.remove' = 1, ### how many clusters to remove every n iterations
      'fast.silhouette.calc' = F ### fast.silhouette.calc = T uses an approximate silhouette calculation (dissimilarity between docs and cluster centers). fast.silhouette.calc = F uses sampled inter-document distances, which can be slower for larger datasets 
      
    )
  }
  
  
  
  ### stem keywords
  seed.words = orig.word.list%>%
    separate_rows(words, sep = ',')%>%
    mutate(words = trimws(words))%>%
    group_by(reasons)%>%
    mutate(reasons2 = str_c(reasons,row_number(),sep='') )%>%
    ungroup()%>%
    separate_rows(words,sep='\\s+' )%>%
    mutate(sign = ifelse(str_detect(words,'^-'),-1,1) )%>%
    mutate(words = str_replace_all(words, c('_'=' ','-'='' ) )  )%>%as.data.frame() 
  
  ### stem only for seeded keywords (unsupervised keywords are already stemmed)
  if(seed.words$type == 'seeded') {
    seed.words$words =  unlist(lapply(seed.words$words%>%tokenize_word_stems(), function(x) str_c(x,collapse = '_') ))
  } else {
    seed.words$words =  unlist(lapply(seed.words$words%>%tokenize_words(), function(x) str_c(x,collapse = '_') ))
  }
  
  print(paste0('words not found in the text: ',
               paste0(seed.words$words[!str_replace_all(seed.words$words,'-','') %in% colnames(dtm)], collapse = ',') ) )
  seed.words = seed.words[str_replace_all(seed.words$words,'-','') %in% colnames(dtm),]
  
  
  word.list = seed.words%>%
    mutate(metric = 1, iteration = 1)%>%
    select(-type)
  
  fnrv = get.reason.vectors(fnoldrv = NULL, fnorigrv = NULL, fnwv, word.list,
                            weigh.by.similarity = T, alpha = 0, beta = 0 ) 
  
  fnorigrv = fnrv
  
  silh = calculate.silhouette(fnrv, fndv,  plot.silhouette = F , 
                              fast.silhouette.calc = control.param$fast.silhouette.calc)
  word.list = word.list%>%
    left_join(silh%>%rename(reasons2 = membership1), by = 'reasons2' )
  
  word.list = word.list%>%
    select(words,sign, reasons, reasons2, metric, iteration, silh, quality)  
  
  
  
  
  ########### iterate
  
  for (iter in seq(2,control.param$num.iter,1)) {
    
    print(paste0('iter: ',iter))
    
    if (iter == 2) {
      print('starting point' )
      print(paste0('unique words: ',length(unique(word.list$words) ) ) )
      print(paste0('unique reasons: ',length(unique( word.list$reasons ) ) ))
      print(paste0('unique reasons2: ',length(unique(str_c(word.list$reasons,word.list$reasons2,sep='') ) ) ))
    }
    
    
    
    
    
    
    cosine.distances = get.metric(fndtm, fnrv, fnwv, fndv,cosine.threshold = 0.5,use.dist = F) 
    
    temp = word.list%>%
      group_by(reasons2, iteration)%>%
      summarize(quality = unique(quality), wordcount = length(words) ) %>%
      ungroup()%>%
      group_by(reasons2)%>%
      summarize(flag = ifelse( max(quality) * control.param$quality.threshold.to.add.keywords >=  last(quality,order_by = iteration)  ,
                               ifelse(max(quality) * control.param$quality.threshold.to.remove.keywords >= last(quality,order_by = iteration), max(last(wordcount,order_by = iteration)-1,0), last(wordcount,order_by = iteration) ),
                               min(last(wordcount,order_by = iteration) + 1 ,control.param$max.keywords.per.topic)        ) )%>%
      ungroup()
    
    
    
    cosine.distances = cosine.distances%>%
      left_join(temp, by = 'reasons2')%>%
      group_by(reasons,reasons2)%>%
      filter( rank(desc(metric), ties.method="first")<= flag  )%>%
      ungroup()
    
    cosine.distances$sign = 1
    
    
    # ########## adjust keyword importance based on conflicting selections in other categories
    # cosine.distances = cosine.distances%>%group_by(words)%>%top_n(1,metric)%>%ungroup()
    
    
    # ######### restrict keywords based on deviation from average metric
    # cosine.distances = cosine.distances%>%
    #   group_by(reasons2)%>%
    #   filter(!(mean(metric)-metric) > 1*sd(metric)  )%>%
    #   ungroup()
    
    
    #############################################################################
    
    fnrv = get.reason.vectors(fnrv,fnorigrv , fnwv,cosine.distances, 
                              weigh.by.similarity = T, alpha = control.param$alpha, beta = control.param$beta  ### alpha changed to 0 from 0.9 temporarily to explore exponential weighting
                              
    )  ## alpha 0 = use new vectors, 1 = pin to old vectors
    
    
    ########## filter similar keywords
    fnrv.temp = as.matrix(fnrv%>%select(starts_with('X')))
    temp = t(sim2(fnrv.temp,fnrv.temp, method = 'cosine', norm = 'l2'))
    temp[lower.tri(temp, diag = FALSE)] <- NA
    temp1 = data.frame(temp)%>%
      mutate(reasonsA = rownames(.))%>%
      melt(id.vars = 'reasonsA', variable.name = 'reasonsB', value.name = 'similarity')%>%
      filter(reasonsA != reasonsB & similarity >=control.param$cluster.merge.threshold )%>%as.data.frame()
    
    for (p in seq_along(temp1$reasonsA)) {
      temp1$reasonsA[temp1$reasonsB == temp1$reasonsB[p]] = temp1$reasonsA[p]
    }
    
    
    remove.list = as.character(unique(temp1$reasonsB)[! unique(temp1$reasonsB) %in% unique(temp1$reasonsA)])
    
 
    silh = calculate.silhouette(fnrv, fndv,  plot.silhouette = F, 
                                fast.silhouette.calc = control.param$fast.silhouette.calc )
    
    cosine.distances = cosine.distances%>%
      left_join(silh%>%rename(reasons2 = membership1), by = 'reasons2' )
    
    if(iter %% control.param$remove.silh.every.niter == 0) {
      temp = as.character(silh%>%filter(silh < control.param$silhouette.threshold)%>%
                            top_n(control.param$silh.clusters.to.remove,desc( silh ))%>%
                            slice(control.param$silh.clusters.to.remove)%>%
                            pull(membership1)
      )
      print(temp)
      remove.list = unique(c(remove.list, temp ) )
    }
    
    cosine.distances = cosine.distances[!cosine.distances$reasons2 %in% remove.list,]
    fnrv = fnrv[!fnrv$reasons2 %in% remove.list,]
    
 
    
    
    #########################################################
    
    
    cosine.distances = cosine.distances%>%
      mutate(iteration = iter)%>%
      select(words,sign,reasons, reasons2, metric, iteration, silh, quality) ## doc.similarity
    
    
    new1 = as.matrix(fnrv%>%select(starts_with('X')))
    orig1 = as.matrix(fnorigrv%>%select(starts_with('X')))
    orig1 = orig1[rownames(orig1) %in% rownames(new1),,drop=F]
    
    ### print progress
    if (iter == 2) {
      cluster.center.progress = data.frame('improvement' = psim2(orig1, new1, method = 'cosine', norm = 'l2'), 'iter' = iter)
      cluster.center.progress$reasons2 = rownames(cluster.center.progress)
      
    } else {
      temp = data.frame('improvement' = psim2(orig1, new1, method = 'cosine', norm = 'l2'),  'iter' = iter)
      temp$reasons2 = rownames(temp) 
      cluster.center.progress = rbind(cluster.center.progress,temp )
      
    }
    
    word.list = rbind(word.list,cosine.distances)
    
    
    temp = word.list%>%filter(iteration == iter)
    print(paste0('unique words: ',length(unique(temp$words) ) )) 
    print(paste0('unique reasons: ',length(unique( temp$reasons ) ) ))
    print(paste0('unique reasons2: ',length(unique(str_c(temp$reasons,temp$reasons2,sep='') ) ) ))
    
    
  }
  
  return.list = list('fndf' = fndf,
                     'fnrv' = fnrv,
                     'fnorigrv' = fnorigrv,
                     'cluster.center.progress' = cluster.center.progress,
                     'word.list' = word.list,
                     'seed.words' = seed.words )
  
  
}

create.rtsne.matrix <- function(fndv, number.of.samples = 10000, seed = 80) {
  set.seed(seed)
  random.indices = sample(seq(1,nrow(fndv),1), min(nrow(fndv),number.of.samples), replace = F )
  rtsne_out <- Rtsne(as.matrix(fndv[random.indices,]), perplexity = 50 ,check_duplicates = FALSE)
  return(list('rtsne_out'=rtsne_out, 'random.indices' = random.indices) )
  
}

plot.cluster.results <- function(rtsne_out,  random.indices, fnlabels ) {
  
  
  
  tsne_plot <- data.frame(x = rtsne_out$Y[,1],
                          y = rtsne_out$Y[,2],
                          desc = fnlabels[random.indices]
  )
  
  
  
  tsne_labels = tsne_plot%>%group_by(desc)%>%summarize(x = median(x), y = median(y))%>%ungroup()
  
  
  g1 = ggplot(tsne_plot%>%sample_n(min(length(random.indices),nrow(tsne_plot)) ), aes(x = x, y = y)) +
    stat_density2d(aes(fill = ..density.., alpha = 1), geom = 'tile', contour = F) +
    scale_fill_distiller(palette = 'Greys') + ###RdYlBu
    geom_point(aes(color=as.factor(as.integer(desc)) ),  size= 3, alpha = 1 ) +
    ggrepel::geom_label_repel(data = tsne_labels,
                              aes(x=x, y=y,label=desc, color = as.factor(as.integer(desc))),
                              fontface = 'bold' ) +
    scale_color_manual(values=rev(hues::iwanthue(200))) +
    theme(legend.position="none")
  
  return(g1)
  
}

predict.clusters <- function(fnkey.sum, fndf, fnwv, fndv, fntokens) {
  
  temp = str_split(fnkey.sum$newwords,',')
  temp =   lapply(temp,function(x) x[x %in% row.names(fnwv)])
  temp1 = str_split(fnkey.sum$weights,',')
  temp1 =   mapply(function(x,y) as.numeric(y[x %in% row.names(fnwv)]),temp,temp1, SIMPLIFY = F)
  newword.avg.vectors = t(mapply(function(x,y) Matrix::colMeans(fnwv[x,,drop=F]*y, na.rm = T) ,
                                 temp,temp1
  ))
  rownames(newword.avg.vectors) = paste0('sim',fnkey.sum$reasons2)
  
  
  
  cos.sim = data.frame( as.matrix(sim2(fndv,newword.avg.vectors , method = c("cosine"), norm = c("l2")) ) )
  
  cos.sim.class = t(apply(cos.sim, 1, function(x) names(x)[order(x, decreasing = T)]))
  cos.sim.class = cos.sim.class[,1:ifelse(ncol(cos.sim)>3,3,ncol(cos.sim)),drop=F]
  colnames(cos.sim.class) = paste0('top',1:ncol(cos.sim.class),'class',sep='')
  
  cos.sim.pred = t(apply(cos.sim, 1, function(x)  sort(x, decreasing = T) ))
  cos.sim.pred = cos.sim.pred[,1:ifelse(ncol(cos.sim)>3,3,ncol(cos.sim)),drop=F]
  colnames(cos.sim.pred) = paste0('top',1:ncol(cos.sim.pred),'pred',sep='')
  
  
  txt_clean =  lapply(fntokens,function(x) str_c(x, collapse = ' ')) 
  txt_clean = lapply(txt_clean, function(x) ifelse(identical(x,character(0)),' ',x) )
  txt_clean = unlist(txt_clean)
  txt_length = unlist(lapply(fntokens,function(x) length(x) ) )
  
  data4 = cbind(fndf,  txt_clean, txt_length, cos.sim.class, cos.sim.pred)
  
  fnkey.sum = fnkey.sum%>%
    left_join(data4%>%mutate(top1class = str_remove_all(top1class,'^sim'))%>%group_by(top1class)%>%
                summarize(counts = n())%>%ungroup()%>%
                rename(reasons2 = top1class), by = 'reasons2' )
  
  return.list = list('predictions' = data4,
                     'keyword.summary' = fnkey.sum,
                     'reason.vectors' = newword.avg.vectors
                     )
  
  return(return.list)
  
}
