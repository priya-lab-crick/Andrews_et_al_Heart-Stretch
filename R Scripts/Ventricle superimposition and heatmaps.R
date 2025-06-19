library(FreqProf)
library(ggplot2)
library(data.table)
library(ggpubr)
library(viridis)
library(dplyr)
library(spatstat)
library(ggbeeswarm)

## Script imports lists of 3D nuclear coordinates classified as CL, TL or delaminating in Imaris, and landmark 
## positions for limits of AP, ML and DV axes. It then infers coordinate positions for nuclei on normalised axes
## enabling superimposition of different ventricles. This is based on a Procrustes superimposition, where each
## axis is aligned and stretched/compressed to a mean length across the population.

##### 1. Import data and plot cell positions on orthogonal axes #####

classifier <- list() # creates a list
list.coords <- dir(pattern = " coords_Detailed.csv") # creates the list of all csv files in the directory

list.comp <- list.coords 
list.comp <- gsub('coords', 'comp_coords', list.coords) ## Get CL coord file list

list.trab <- list.comp
list.trab <- gsub('comp', 'trab', list.trab) ## Get TL coords file list

list.delam <- list.comp
list.delam <- gsub('comp', 'delam', list.delam) ## Get DL coords file list

for (k in 1:length(list.coords)){

coords <- read.csv(list.coords[[k]], skip = 3) ## Get landmark coordinates per embryo

coords <- coords[,1:3] ## Ignore non-coordinate info

coords$Position <- c('OFT', 'Apex', 'Front', 'Back', 'Valve', 'OC') ## Classify landmarks
coords$Axis <- c('Top-Bottom', 'Top-Bottom', 'Front-Back', 'Front-Back', 'Left-Right', 'Left-Right') ## Classify axes

TB <- subset(coords, Axis=='Top-Bottom') ## Subset axes
TB <- TB[,1:4]
FB <- subset(coords, Axis=='Front-Back')
FB <- FB[,1:4]
LR <- subset(coords, Axis=='Left-Right')
LR <- LR[,1:4]

### Build top-bottom axis ###

X_ext <- TB[1,1] + (TB[1,1] - TB[2,1]) ## Extrapolate axis limits (extended) above and below (minimised)
Y_ext <- TB[1,2] + (TB[1,2] - TB[2,2])
Z_ext <- TB[1,3] + (TB[1,3] - TB[2,3])

X_min <- TB[2,1] - (TB[1,1] - TB[2,1])
Y_min <- TB[2,2] - (TB[1,2] - TB[2,2])
Z_min <- TB[2,3] - (TB[1,3] - TB[2,3])

TB_long <- TB ## Build expanded reference frame
TB_long[1, 1] <- X_ext
TB_long[1, 2] <- Y_ext
TB_long[1, 3] <- Z_ext
TB_long[2, 1] <- X_min
TB_long[2, 2] <- Y_min
TB_long[2, 3] <- Z_min

## Calculate axis length
TB_d <- sqrt((TB_long[1, 1] - TB_long[2, 1])^2 + (TB_long[1, 2] - TB_long[2, 2])^2 + (TB_long[1, 3] - TB_long[2, 3])^2)

## Round to nearest integer
TB_d <- as.numeric(format(round(TB_d, 0), nsmall = 0))

## Interpolate coordinates at 1 um distance
TB_2 <- approxm(TB_long[,1:3], TB_d, method = "linear")

## Classify new coordinates by distance along axis, starting from 0
TB_2$cumdist <- 0 

for (i in 2:nrow(TB_2)) { ## For loop classifies each row as previous + 1
  
  m <- TB_2[i-1,]$cumdist + TB_d / nrow(TB_2)
  
  TB_2$cumdist[i] <- m
  
}

TB_2$Normdist <- TB_2$cumdist - min(TB_2$cumdist)

ggplot(TB_2, aes(x = Position.X, y = Position.Y, colour = Normdist)) + ## Plot axis
  geom_point()

### Build front-back axis ###

X_ext <- FB[1,1] + (FB[1,1] - FB[2,1]) ## Repeat above steps for FB axis
Y_ext <- FB[1,2] + (FB[1,2] - FB[2,2])
Z_ext <- FB[1,3] + (FB[1,3] - FB[2,3])

X_min <- FB[2,1] - (FB[1,1] - FB[2,1])
Y_min <- FB[2,2] - (FB[1,2] - FB[2,2])
Z_min <- FB[2,3] - (FB[1,3] - FB[2,3])

FB_long <- FB
FB_long[1, 1] <- X_ext
FB_long[1, 2] <- Y_ext
FB_long[1, 3] <- Z_ext
FB_long[2, 1] <- X_min
FB_long[2, 2] <- Y_min
FB_long[2, 3] <- Z_min

FB_d <- sqrt((FB_long[1, 1] - FB_long[2, 1])^2 + (FB_long[1, 2] - FB_long[2, 2])^2 + (FB_long[1, 3] - FB_long[2, 3])^2)

FB_d <- as.numeric(format(round(FB_d, 0), nsmall = 0))

FB_2 <- approxm(FB_long[,1:3], FB_d, method = "linear")

FB_2$cumdist <- 0

for (i in 2:nrow(FB_2)) {
  
  m <- FB_2[i-1,]$cumdist + FB_d / nrow(FB_2)
  
  FB_2$cumdist[i] <- m
  
}

FB_2$Normdist <- FB_2$cumdist - min(FB_2$cumdist)

ggplot(FB_2, aes(x = Position.X, y = Position.Y, colour = Normdist)) +
  geom_point()

### Build left-right axis ###

X_ext <- LR[1,1] + (LR[1,1] - LR[2,1]) ## Repeat above steps for ML axis
Y_ext <- LR[1,2] + (LR[1,2] - LR[2,2])
Z_ext <- LR[1,3] + (LR[1,3] - LR[2,3])

X_min <- LR[2,1] - (LR[1,1] - LR[2,1])
Y_min <- LR[2,2] - (LR[1,2] - LR[2,2])
Z_min <- LR[2,3] - (LR[1,3] - LR[2,3])

LR_long <- LR
LR_long[1, 1] <- X_ext
LR_long[1, 2] <- Y_ext
LR_long[1, 3] <- Z_ext
LR_long[2, 1] <- X_min
LR_long[2, 2] <- Y_min
LR_long[2, 3] <- Z_min

LR_d <- sqrt((LR_long[1, 1] - LR_long[2, 1])^2 + (LR_long[1, 2] - LR_long[2, 2])^2 + (LR_long[1, 3] - LR_long[2, 3])^2)

LR_d <- as.numeric(format(round(LR_d, 0), nsmall = 0))

LR_2 <- approxm(LR_long[,1:3], LR_d, method = "linear")

LR_2$cumdist <- 0

for (i in 2:nrow(LR_2)) {
  
  m <- LR_2[i-1,]$cumdist + LR_d / nrow(LR_2)
  
  LR_2$cumdist[i] <- m
  
}

LR_2$Normdist <- LR_2$cumdist - min(LR_2$cumdist)

ggplot(LR_2, aes(x = Position.X, y = Position.Y, colour = Normdist)) +
  geom_point()

### Import nuclear coordinates ###

comp <- read.csv(list.comp[[k]], skip = 3) ## Get CL coordinates
comp <- comp[,1:3]
comp$type <- 'comp'

if (file.exists(list.trab[[k]])) { ## Get TL coordinates (if any)
  trab <- read.csv(list.trab[[k]], skip = 3)
  trab <- trab[,1:3]
  trab$type <- 'trab'
  
}

if (file.exists(list.delam[[k]])) { ## Get DL coordinates
  delam <- read.csv(list.delam[[k]], skip = 3)
  delam <- delam[,1:3]
  delam$type <- 'delam'
  
}

comp <- rbind(get0("trab"),get0("delam"), get0("comp")) ## Merge (if they exist)

TB_values <- rep(0, nrow(TB_2)) ## Empty vector for top-bottom values

comp_values <- rep(0, nrow(comp)) 

for(p in 1:nrow(comp)) {
  
  for(i in 1:nrow(TB_2)) {
    
    #Calculate distance values to specific point for all coordinates making up TB_2 element
    o <- sqrt(abs(TB_2[i,]$Position.X - comp[p,]$Position.X)^2 
              + abs(TB_2[i,]$Position.Y - comp[p,]$Position.Y)^2
              + abs(TB_2[i,]$Position.Z - comp[p,]$Position.Z)^2)
    
    #List distance values in new vector
    TB_values[i] <- o
    
    #For each point, find the closest coordinate and list its row value in new vector
    comp_values[p] <- which.min(TB_values)
    
    #Filter coordinate points of TB_2 for those corresponding to values in point vector 
    TB_coord <- TB_2[comp_values,]
    
  }}

FB_values <- rep(0, nrow(FB_2)) ## Empty vector for front-back values

comp_values <- rep(0, nrow(comp))

for(p in 1:nrow(comp)) {
  
  for(i in 1:nrow(FB_2)) {
    
    #Calculate distance values to specific point for all coordinates making up FB_2 element
    o <- sqrt(abs(FB_2[i,]$Position.X - comp[p,]$Position.X)^2 
              + abs(FB_2[i,]$Position.Y - comp[p,]$Position.Y)^2
              + abs(FB_2[i,]$Position.Z - comp[p,]$Position.Z)^2)
    
    #List distance values in new vector
    FB_values[i] <- o
    
    #For each point, find the closest coordinate and list its row value in new vector
    comp_values[p] <- which.min(FB_values)
    
    #Filter coordinate points of FB_2 for those corresponding to values in point vector 
    FB_coord <- FB_2[comp_values,]
    
  }}

LR_values <- rep(0, nrow(LR_2)) ## Empty vector for left-right values

comp_values <- rep(0, nrow(comp))

for(p in 1:nrow(comp)) {
  
  for(i in 1:nrow(LR_2)) {
    
    #Calculate distance values to specific point for all coordinates making up LR_2 element
    o <- sqrt(abs(LR_2[i,]$Position.X - comp[p,]$Position.X)^2 
              + abs(LR_2[i,]$Position.Y - comp[p,]$Position.Y)^2
              + abs(LR_2[i,]$Position.Z - comp[p,]$Position.Z)^2)
    
    #List distance values in new vector
    LR_values[i] <- o
    
    #For each point, find the closest coordinate and list its row value in new vector
    comp_values[p] <- which.min(LR_values)
    
    #Filter coordinate points of LR_2 for those corresponding to values in point vector 
    LR_coord <- LR_2[comp_values,]
    
  }}

comp_coords <- data.frame(matrix(ncol = 3, nrow = nrow(TB_coord))) ## Compile all normalised coordinates
x <- c("TB", "FB", "LR")
colnames(comp_coords) <- x

comp_coords$TB <- TB_coord$Normdist
comp_coords$FB <- FB_coord$Normdist
comp_coords$LR <- LR_coord$Normdist

c <- cbind(comp, comp_coords)

compact <- subset(c, type=='comp')
trabecular <- subset(c, type=='trab')
delaminating <- subset(c, type=='delam')

c$Embryo <- k ## Embryo ID

c$Identity <- list.coords[[k]] ## File name

classifier[[k]] <- c

if (exists('trab')) { ## Delete temporary files
  
  remove(trab)
  
}
  
if (exists('comp')) {
  
  remove(comp)
  
}

if (exists('delam')) {
  
  remove(delam)
  
}}

for (i in 1:length(classifier)){

classifier[[i]]$RedTB <- classifier[[i]]$TB - min(classifier[[i]]$TB) ## Min-max normalise axis lengths
classifier[[i]]$NormTB <- classifier[[i]]$RedTB / max(classifier[[i]]$RedTB)
classifier[[i]]$RedFB <- classifier[[i]]$FB - min(classifier[[i]]$FB)
classifier[[i]]$NormFB <- classifier[[i]]$RedFB / max(classifier[[i]]$RedFB)
classifier[[i]]$RedLR <- classifier[[i]]$LR - min(classifier[[i]]$LR)
classifier[[i]]$NormLR <- classifier[[i]]$RedLR / max(classifier[[i]]$RedLR)

}

##### 2. Classify stages and scale to normalised axes #####

p <- rbindlist(classifier)

p$NormFB <- 1 - p$NormFB

S48hpf <- p[grep("48hpf", p$Identity), ]
S48hpf$hpf <- 48

d <- aggregate(S48hpf$RedTB, by = list(S48hpf$Embryo), max)
mean(d[,2])

e <- aggregate(S48hpf$RedLR, by = list(S48hpf$Embryo), max)
mean(e[,2])

f <- aggregate(S48hpf$RedFB, by = list(S48hpf$Embryo), max)
mean(f[,2])

S48hpf$scaled_TB <- S48hpf$NormTB * mean(d[,2])
S48hpf$scaled_LR <- S48hpf$NormLR * mean(e[,2])
S48hpf$scaled_FB <- S48hpf$NormFB * mean(f[,2])

S60hpf <- p[grep("60hpf", p$Identity), ]
S60hpf$hpf <- 60

d <- aggregate(S60hpf$RedTB, by = list(S60hpf$Embryo), max)
mean(d[,2])

e <- aggregate(S60hpf$RedLR, by = list(S60hpf$Embryo), max)
mean(e[,2])

f <- aggregate(S60hpf$RedFB, by = list(S60hpf$Embryo), max)
mean(f[,2])

S60hpf$scaled_TB <- S60hpf$NormTB * mean(d[,2])
S60hpf$scaled_LR <- S60hpf$NormLR * mean(e[,2])
S60hpf$scaled_FB <- S60hpf$NormFB * mean(f[,2])

S72hpf <- p[grep("72hpf", p$Identity), ]
S72hpf$hpf <- 72

d <- aggregate(S72hpf$RedTB, by = list(S72hpf$Embryo), max)
mean(d[,2])

e <- aggregate(S72hpf$RedLR, by = list(S72hpf$Embryo), max)


mean(e[,2])

f <- aggregate(S72hpf$RedFB, by = list(S72hpf$Embryo), max)
mean(f[,2])

S72hpf$scaled_TB <- S72hpf$NormTB * mean(d[,2])
S72hpf$scaled_LR <- S72hpf$NormLR * mean(e[,2])
S72hpf$scaled_FB <- S72hpf$NormFB * mean(f[,2])

S84hpf <- p[grep("84hpf", p$Identity), ]
S84hpf$hpf <- 84

d <- aggregate(S84hpf$RedTB, by = list(S84hpf$Embryo), max)
mean(d[,2])

e <- aggregate(S84hpf$RedLR, by = list(S84hpf$Embryo), max)

mean(e[,2])

f <- aggregate(S84hpf$RedFB, by = list(S84hpf$Embryo), max)
mean(f[,2])

S84hpf$scaled_TB <- S84hpf$NormTB * mean(d[,2])
S84hpf$scaled_LR <- S84hpf$NormLR * mean(e[,2])
S84hpf$scaled_FB <- S84hpf$NormFB * mean(f[,2])

S96hpf <- p[grep("96hpf", p$Identity), ]
S96hpf$hpf <- 96

d <- aggregate(S96hpf$RedTB, by = list(S96hpf$Embryo), max)
mean(d[,2])

e <- aggregate(S96hpf$RedLR, by = list(S96hpf$Embryo), max)

mean(e[,2])

f <- aggregate(S96hpf$RedFB, by = list(S96hpf$Embryo), max)
mean(f[,2])

S96hpf$scaled_TB <- S96hpf$NormTB * mean(d[,2])
S96hpf$scaled_LR <- S96hpf$NormLR * mean(e[,2])
S96hpf$scaled_FB <- S96hpf$NormFB * mean(f[,2])


S108hpf <- p[grep("108hpf", p$Identity), ]
S108hpf$hpf <- 108

d <- aggregate(S108hpf$RedTB, by = list(S108hpf$Embryo), max)
mean(d[,2])

e <- aggregate(S108hpf$RedLR, by = list(S108hpf$Embryo), max)

mean(e[,2])

f <- aggregate(S108hpf$RedFB, by = list(S108hpf$Embryo), max)
mean(f[,2])

S108hpf$scaled_TB <- S108hpf$NormTB * mean(d[,2])
S108hpf$scaled_LR <- S108hpf$NormLR * mean(e[,2])
S108hpf$scaled_FB <- S108hpf$NormFB * mean(f[,2])

S120hpf <- p[grep("120hpf", p$Identity), ]
S120hpf$hpf <- 120

d <- aggregate(S120hpf$RedTB, by = list(S120hpf$Embryo), max)
mean(d[,2])

e <- aggregate(S120hpf$RedLR, by = list(S120hpf$Embryo), max)

mean(e[,2])

f <- aggregate(S120hpf$RedFB, by = list(S120hpf$Embryo), max)
mean(f[,2])

S120hpf$scaled_TB <- S120hpf$NormTB * mean(d[,2])
S120hpf$scaled_LR <- S120hpf$NormLR * mean(e[,2])
S120hpf$scaled_FB <- S120hpf$NormFB * mean(f[,2])

p <- rbind(S48hpf, S60hpf, S72hpf, S84hpf, S96hpf, S108hpf, S120hpf)

p$Embryo <- as.factor(p$Embryo)

write.csv(p, 'Ventricle morphometrics.csv')

## OR ##

p <- read.csv('Ventricle morphometrics.csv')

##### 3. Plot superimposed data #####

ggplot(p, aes(scaled_LR, scaled_TB)) +
  stat_density2d(aes(fill = ..level..), geom = "polygon", h = 40) +
  stat_density2d(data = subset(p, type=='delam'), aes(fill = ..level..), contour_var = "count", geom = "polygon", h = 25) +
  scale_fill_viridis(option = 'inferno') +
  theme_classic() +
  coord_fixed() +
  xlim(-10, 105) +
  ylim(190, -20) +
  facet_wrap(~hpf, nrow = 1) +
  ylab(bquote('Distance from OFT (µm)')) +
  xlab('Left-Right Position (µm)') 

###################################################################


