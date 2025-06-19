library(alphashape3d)
library(Rvcg)
library(plotly)

## Reads in processed data from 'Ventricle superimposition and heatmaps.R' script, ie. a data frame containing raw and superimposed nuclear coordinates

p <- read.csv('20211120_Ventricle morphometrics.csv') ## Read in coordinates
 
##### Plot single embryo alpha hull #####

single <- subset(p, Embryo == 50,)

points <- single[,18:20]

plot_ly(mtcars, x = ~wt, y = ~hp, z = ~qsec, color = ~am, colors = c('#BF382A', '#0C4B8E'))

colnames(points) <- c('x', 'y', 'z')

matrix <- as.matrix(points)

shape <- ashape3d(matrix, alpha = 80, pert = FALSE, eps = 1e-09)

mesh <- as.mesh3d(shape, smooth = FALSE)

clean.mesh <- vcgClean(mesh, sel = c(1:7), tol = 2, silent = FALSE, iterate = FALSE)

smooth.mesh <- vcgSmooth(clean.mesh, type = c("taubin"), iteration = 10, lambda = 0.5, mu = -0.53, delta = 0.1)

remesh <- vcgIsotropicRemeshing(smooth.mesh, TargetLen = 1)

plot3d(remesh, col = 'red', alpha = 1, front = 'lines', main = "", sub = "", ann = FALSE, xlab = "AP", ylab = "ML", zlab = "DV")

aspect3d("iso")

rgl.snapshot('3dplot_volume_side.png', fmt = 'png')

plot3d(points, size = 5)

aspect3d("iso")

rgl.postscript('3dplot_volume_side.svg', fmt = "svg", drawText = TRUE)

aspect3d("iso")

plot(shape, indexAlpha = "all")

##### Batch processing #####

all_areas <- rep(0, length(unique(p$Embryo)))
all_volumes <- rep(0, length(unique(p$Embryo)))

for (i in 1:length(unique(p$Embryo))) {
  
  nuclei <- subset(p, Embryo==i & type=='comp')
  
  points <- nuclei[,18:20]
  
  colnames(points) <- c('x', 'y', 'z')
  
  matrix <- as.matrix(points)
  
  shape.raw <- ashape3d(matrix, alpha = 70, pert = TRUE, eps = 1e-09) # calculate alpha hull
  
  mesh.raw <- as.mesh3d(shape.raw, smooth = TRUE) # repair mesh, clean, smooth, uniform remesh
  
  shape <- vcgClean(mesh.raw, sel = c(1:7), tol = 2, silent = FALSE, iterate = FALSE)
  
  shape <- vcgIsotropicRemeshing(shape, TargetLen = 2)
  
  shape <- vcgSmooth(shape, type = c("taubin"),
                     iteration = 50,
                     lambda = 0.5,
                     mu = -0.53,
                     delta = 0.1)
  
  shape <- vcgIsotropicRemeshing(shape, TargetLen = 2)
  
  all_volumes[i] <- vcgVolume(shape) # calculate area, store in vector
  
  all_areas[i] <- vcgArea(shape) # calculate area, store in vector
  
}

stages <- unique(p[,c(9, 17)]) ## Get unique embryo IDs for each hpf

stages$area <- all_areas ## Append surface area values from batch
stages$volume <- all_volumes ## Append volume values from batch

ggplot(stages[!stages$hpf == 60,], aes(x = as.factor(hpf), y = area)) + ## Plot CL surface area from 72 hpf
  geom_jitter(width = 0.2, colour = 'grey', size = 0.2) +
  geom_boxplot(outlier.shape = NA, colour = 'black', size = 0.1, alpha = 0, width = 0.5) +
  xlab("hpf") +
  theme_bw() +
  ylab(bquote('Compact layer area '(µm^2))) +
  theme(legend.position = "none", 
        axis.text = element_text(size = 7), 
        axis.title = element_text(size = 7), 
        axis.text.x =element_text(colour="black"),
        axis.text.y =element_text(colour="black")) +
  ylim(25000, 55000)

ggsave(file="20240311 Ventricle Areas.svg", width=3.8, height=4, units = "cm")

ggplot(stages[!stages$hpf == 60,], aes(x = as.factor(hpf), y = volume/1000000)) + ## Plot CL volume from 72 hpf (divide by 1,000,000 to convert from um3 to nl)
  geom_jitter(width = 0.2, colour = 'grey', size = 0.2) +
  geom_boxplot(outlier.shape = NA, colour = 'black', size = 0.1, alpha = 0, width = 0.5) +
  xlab("hpf") +
  theme_bw() +
  ylab('Ventricle volume (nl)') +
  theme(legend.position = "none", 
        axis.text = element_text(size = 7), 
        axis.title = element_text(size = 7), 
        axis.text.x =element_text(colour="black"),
        axis.text.y =element_text(colour="black")) +
  ylim(0.3, 1)

ggsave(file="20240311 Ventricle Volumes.svg", width=3.4, height=4, units = "cm")

