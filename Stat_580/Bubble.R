library(plotly)

# read in data
df <- read.csv('D:\\DataspellProjects\\MAS_Fall_22\\Stat_580\\data\\auto.csv')
str(df)

# mpg vs weights with hp as size and cyl factor
p <- ggplot(data = df, aes(y = mpg, x = wt, size = hp, col = factor(cyl))) +
	geom_point(alpha = 0.7) +
	scale_size(range = c(.5, 15), name='Gross Horsepower') +
	ggtitle('MPG vs. Weights')


ggplotly(p)
save_image(p, 'bubble-plot.png')