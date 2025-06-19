library(data.table)
library(dplyr)
library(ggplot2)
library(viridis)

## Script reads in normalised (scaled min-max in each channel, 0-255) signal intensities for CFP, YFP and RFP. Values
## are recorded sequentially for all ROIs, such with all CFP first, then YFP then RFP.

list.cols <- dir(pattern = "RGB.csv") # creates the list of all the csv files in the directory

big.ridge.list <- list()

clone.counts <- NULL

for (b in 1:length(list.cols)) { 
  
  d <- read.csv(list.cols[b])
  
  Y <- d[1:(nrow(d)/3),] ## Filter out first colour set
  
  B <- d[(nrow(d)/3 + 1):(nrow(d)/3 + nrow(d)/3),] ## Second colour set
  
  R <- d[(2 * (nrow(d)/3) + 1):(nrow(d)),] ## Third colour set
  
  RGB <- cbind(Y, B[,2], R[,2]) ## Merge values for each cell side-by-side
  
  colnames(RGB) <- c('Cell', 'Y', 'Group', 'B', 'R') 
  
  ctrl <- RGB[RGB$Group == 0,] ## Filter out control ROIs (Cre- non-myocardial tissue)
  
  ctrl.dists <- dist(as.matrix(ctrl[,c(2, 4, 5)]), method = "euclidean", diag = FALSE, upper = FALSE, p = 2) ## Measure colour distances between ctrl ROIs
  
  Ridges <- RGB[RGB$Group > 0,] ## Filter out ridges
  
  Ridges$clone <- 0 ## Set clone IDs to 0
  
  clones <- data.frame(matrix(nrow = length(unique(Ridges$Group)), ncol = 5)) ## Build new dataframe for clone ID classification
  colnames(clones) <- c('Heart', 'Ridge', 'Cells', 'Clones', 'Diversity') ## Name columns
    
  all.ridges <- list()
    
  for (g in unique(Ridges$Group)) {
      
    ridge <- Ridges[Ridges$Group == g,] # Select ridge
      
    ridge$clone <- 0 # Give default clone ID as 0 to all columns
      
    c <- 0 # Reset clone count
      
    remainder <- ridge # Make new dataframe with all cells to be given a clone ID (all of them)   IS THIS NEEDED?
      
    while (nrow(ridge[ridge$clone == 0,]) > 1) { # New for loop defines clone IDs
        
       c <- c + 1 # Move clone ID up 1 value for each loop
        
       undefined <- ridge[ridge$clone == 0,] # Get all cells without a clone ID
        
       Ridge.dists <- dist(as.matrix(undefined[,c(2, 4, 5)]), method = "euclidean", diag = FALSE, upper = FALSE, p = 2) # Get euclidean distances between all cells 
        
       Ridge.dists <- as.matrix(Ridge.dists) # Make it a matrix
        
       Ridge.dists <- Ridge.dists[-1,1] # Get distances from first cell to all other cells (First row, minus first column)
        
       cell <- undefined[1,] # Get the cell of interest from undefined list
        
       Cell.ID <- cell$Cell # Get cell ID for cell of interest
        
       ridge[ridge$Cell == Cell.ID,]$clone <- c # Give cell of interest a unique clone ID in main dataframe
        
       remainder <- undefined[-1,] # Make new dataframe for remaining unclassified cells 
        
      clone.search <- Ridge.dists <= max(ctrl.dists) #Finds cells of the same clone in TRUE/FALSE vector
        
       for (i in 1:length(clone.search)) { 
          
         if (clone.search[i] == TRUE) { # Asks if each remaining cell is within same clone of cell of interest
            
          ID2 <- remainder[i,]$Cell # If so, gets ID of that cell in new 'ID2' value
            
          ridge[ridge$Cell == ID2,]$clone <- c # And nominates same clone ID as cell of interest
            
         }
          
       }
        
     }
      
     all.ridges[[g]] <- ridge
      
     clones[g, 2] <- g
     clones[g, 3] <- nrow(ridge)
     clones[g, 4] <- length(unique(ridge$clone))
     clones[g, 5] <- length(unique(ridge$clone)) / nrow(ridge)
     clones[g, 1] <- b
      
     clone.count <- ridge %>% count(clone) 
     clone.count$size <- nrow(ridge)
      
     clone.counts <- rbind(clone.count, if(exists("clone.counts")) clone.counts)
      
   }
    
   big.ridge.list[[b]] <- clones # Adds clone data for whole heart to main dataframe across Embryos 
    
 }


ridge.list <- rbindlist(big.ridge.list)

ridge.list <- ridge.list[rowSums(is.na(ridge.list)) == 0, ] 

ridge.list$Size <- ridge.list$Cells / ridge.list$Clones

ggplot(ridge.list[ridge.list$Cells < 8,], aes(x = Cells, y = Clones)) + ## Plot ridge size vs unique clones
  geom_point(size = 1, stroke = 0.2, alpha = 0.2) +
  scale_colour_viridis() +
  coord_fixed() +
  theme_bw() +
  theme(legend.position = "none", 
        axis.text = element_text(size = 7, family = "Arial"), 
        axis.title = element_text(size = 7, family = "Arial"), 
        axis.text.x =element_text(size = 6, colour="black", family = "Arial"),
        axis.text.y =element_text(size = 6, colour="black", family = "Arial"),
        panel.grid.minor = element_line(size = 0.1),
        panel.grid.major = element_line(size = 0.3)) +
  geom_smooth(method = 'lm', linetype = 'dashed', size = 0.2, colour = 'black') +
  xlim(1, 7) +
  ylim(1, 7) +
  ylab('Number of clones') +
  xlab('Number of cells')

ggplot(clone.counts, aes(x = size, y = n)) + ## Plot ridge size vs clone size
  geom_point(size = 1, stroke = 0.2, alpha = 0.2) +
  scale_colour_viridis() +
  coord_fixed() +
  theme_bw() +
  theme(legend.position = "none", 
        axis.text = element_text(size = 7, family = "Arial"), 
        axis.title = element_text(size = 7, family = "Arial"), 
        axis.text.x =element_text(size = 6, colour="black", family = "Arial"),
        axis.text.y =element_text(size = 6, colour="black", family = "Arial"),
        panel.grid.minor = element_line(size = 0.1),
        panel.grid.major = element_line(size = 0.3)) +
  geom_smooth(method = 'lm', linetype = 'dashed', size = 0.2, colour = 'black') +
  xlim(1, 7) +
  ylim(1, 7) +
  ylab('Clone size') +
  xlab('Number of cells')

