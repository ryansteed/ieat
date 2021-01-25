library(tidyverse)
library(ggforestplot)
library(dmetar)
library(comprehenr)
library(grid)

# df_linear <-
#   ggforestplot::df_linear_associations %>%
#   dplyr::arrange(name) %>%
#   dplyr::filter(dplyr::row_number() <= 30)
# 
# # Forestplot
# forestplot(
#   df = df_linear,
#   estimate = beta,
#   logodds = FALSE,
#   colour = trait,
#   title = "Associations to metabolic traits",
#   xlab = "1-SD increment in cardiometabolic trait
#   per 1-SD increment in biomarker concentration"
# )

df = read.csv("output/results.csv") %>% 
  mutate_if(is.factor, as.character) %>%
  mutate_if(is.numeric, as.double) %>%
  as_tibble() %>%
  group_by(Test) %>%
  mutate(p_min = min(p)) %>%
  ungroup()
df$se = se.from.p(effect.size = df$d, p=df$p, N=df$N_a, effect.size.type="difference")$StandardError
df$sig = factor(
  ifelse(df$p_min < 0.1, "bold(Significant)*phantom(' ')*(italic(p) < 0.1)", "bold(Insignificant)"),
  levels=c("bold(Significant)*phantom(' ')*(italic(p) < 0.1)", "bold(Insignificant)")
)

cbp1 <- c("#56B4E9", "#009E73",
          "#0072B2", "#CC79A7")
plot = forestplot(
  df = df,
  name = Test,
  estimate = d,
  pvalue = p,
  psignif = 0.1,
  ci= 0.9,
  colour = Model,
  xlab = "Effect Size (d)",
  ylab = "Bias Test (iEAT)",
  size=50,
  label = "label_parsed"
) +
  ggforce::facet_row(
    facets = ~sig,
    label = "label_parsed",
    scales = "free_y",
    space = "free"
  ) +
  scale_colour_manual(values=cbp1[3:4]) + 
  geom_text(
    label=paste0("N=",df$N_a+df$N_t),
    x=-7.5,
    vjust=2.5,
    size=3,
    fontface="italic"
  ) +
  coord_cartesian(clip='off')
ggsave("output/results-abbrv.png", plot=plot, height=4, width=8)

