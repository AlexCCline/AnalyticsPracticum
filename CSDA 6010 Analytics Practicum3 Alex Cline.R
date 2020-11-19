#__________________________Market Basket Analytics______________________________
#load dataset
CCSell <- read.csv("~/CSDA 5330 Alex Cline/CatalogCrossSell.csv")
View(CCSell)
#libraries
library(arulesViz)
library(plotly)
library(arules)
library(yaml)
library(colorspace)
library(dplyr)
glimpse(CCSell)
#_____________________________Data Preparation___________________________________

#drop Customer.Number
drop <- c("Customer.Number") 
CCSell = CCSell[,!(names(CCSell) %in% drop)]

#matrix
CCSell.mat <- as.matrix(CCSell)
#transactions
CCSell.trans <- as(CCSell.mat, "transactions")
#plot item frequency
itemFrequencyPlot(CCSell.trans)

#_____________________________Model Selection_____________________________________

#apriori algorithm
CCSell.rules<-apriori(CCSell.trans, parameter = list(support = 0.1, confidence = 0.1, target = "rules"))
inspect(CCSell.rules)
summary(CCSell.rules)

#Cg%rules
Cg0.05rules<-CCSell.rules[quality(CCSell.rules)$confidence>0.05]
inspect(Cg0.05rules)
Cg0.5rules<-CCSell.rules[quality(CCSell.rules)$confidence>0.5]
inspect(Cg0.5rules)
Cg0.3rules<-CCSell.rules[quality(CCSell.rules)$confidence>0.3]
inspect(Cg0.3rules)

#Get top 10 lift rules
Top.10.lift.Rules<-sort(CCSell.rules, decreasing = TRUE, na.last = NA, by = "lift")
inspect(head(Top.10.lift.Rules, 10))

#weighted algorithm
CCSell.weighted <- weclat(CCSell.trans, parameter = list(support = 0.1, confidence = 0.1))
inspect(sort(CCSell.weighted))
summary(CCSell.weighted)

#rule induction for weighted
CCSell.ruleind <- ruleInduction(CCSell.weighted, confidence = 0.1)
inspect(sort(CCSell.ruleind))
inspect(sort(head(CCSell.ruleind, 10)))

#eclat algorithm
CCSell.eclat <- eclat(CCSell.trans)
inspect(sort(CCSell.eclat))
summary(CCSell.eclat)

#rule induction for eclat
CCSell.eclatind <- ruleInduction(CCSell.eclat, confidence = 0.1)
inspect(sort(CCSell.eclatind))
inspect(sort(head(CCSell.eclatind, 10)))
#___________________________Visualization_________________________________________

#apirori algorithm plots
plot(CCSell.rules, control = list(col=sequential_hcl(100)))
plot(CCSell.rules, method = "two-key plot")
plot(CCSell.rules, method = "paracoord", control = list(reorder = TRUE))
#creating interactive plot
plot(CCSell.rules, method = "graph", engine = "htmlwidget")
#apriori rules in excel
write.csv(inspect(CCSell.rules), file = ("CatalogCrossSell1.csv"))

#weighted rule induction algorithm plots
plot(CCSell.weighted, control = list(col=sequential_hcl(100)))
plot(CCSell.weighted, method = "paracoord", control = list(reorder = TRUE))
#creating interactive plot for rule induction
plot(CCSell.ruleind, method = "graph", engine = "htmlwidget")
#weighted rules in excel
write.csv(inspect(CCSell.weighted), file = ("CatalogCrossSell2.csv"))

#eclat algorithm plots
plot(CCSell.eclat, control = list(col=sequential_hcl(100)))
plot(CCSell.eclat, method = "paracoord", control = list(reorder = TRUE))
#creating interactive plot for eclat rule induction
plot(CCSell.eclatind, method = "graph", engine = "htmlwidget")
#eclat rules in excel
write.csv(inspect(CCSell.eclat), file = ("CatalogCrossSell3.csv"))

#_____________________________Clustering__________________________________________

#cluster items with 5% support
s <- CCSell.trans[,itemFrequency(CCSell.trans)>0.05]
d_jaccard <- dissimilarity(s, which = "items")
plot(hclust(d_jaccard, method = "ward.D2"), main = "Dendrogram for items")


##  calculate Jaccard distances and do hclust
d_jaccard2 <- dissimilarity(s)
hc <- hclust(d_jaccard2, method = "ward.D2")
plot(hc, labels = FALSE, main = "Dendrogram for Transactions (Jaccard)")


## get 20 clusters and look at the difference of the item frequencies (bars) 
## for the top 20 items) in cluster 1 compared to the data (line) 
assign <- cutree(hc, 20)
itemFrequencyPlot(s[assign==1], population=s, topN=20)

#5% sample of the dataset (for easier visualization)
v <- sample(CCSell.trans, 250) 
## calculate affinity-based distances between transactions and do hclust
d_affinity <- dissimilarity(v, method = "affinity")
hc <- hclust(d_affinity, method = "ward.D2")
plot(hc, labels = FALSE, main = "Dendrogram for Transactions (Affinity)")


## cluster association rules
rules <- apriori(v, parameter=list(support=0.1))
rules <- subset(rules, subset = support > 0.2)
inspect(rules)


## use affinity to cluster rules using sample data
d_affinity <- dissimilarity(rules, method = "affinity", 
                            args = list(transactions = v))
hc <- hclust(d_affinity, method = "ward.D2")
plot(hc, main = "Dendrogram for Rules (Affinity)") 

## create 4 groups and inspect the rules in the first group.
assign <- cutree(hc, k = 3)
inspect(rules[assign == 1])
inspect(rules[assign == 2])
inspect(rules[assign == 3])




