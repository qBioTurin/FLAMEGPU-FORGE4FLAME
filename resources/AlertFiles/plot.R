library(ggplot2)
library(dplyr)

# Define file names and corresponding scenario names
file_names <- c("percentage_infected_by_day_explosion.csv",
                "percentage_infected_by_day_20_chirur.csv",
                "percentage_infected_by_day_40_chirur.csv",
                "percentage_infected_by_day_80_chirur.csv",
                "percentage_infected_by_day_20_ffp2.csv",
                "percentage_infected_by_day_40_ffp2.csv",
                "percentage_infected_by_day_80_ffp2.csv")

# Initial scenario names as read from files
scenario_names <- c("Baseline",
                    "Surgical 20%",
                    "Surgical 40%",
                    "Surgical 80%",
                    "FFP2 20%",
                    "FFP2 40%",
                    "FFP2 80%")

# Read each CSV, add the scenario column, and combine into one data frame
all_data <- lapply(seq_along(file_names), function(i) {
  df <- read.csv(file_names[i])
  df$Scenario <- scenario_names[i]
  df
}) %>% bind_rows()

# Reorder the factor levels to match the desired order:
# Baseline, Surgical 20%, Surgical 40%, FFP2 20%, FFP2 40%, Surgical 80%, FFP2 80%
desired_order <- c("Baseline", "Surgical 20%", "Surgical 40%", "Surgical 80%",
                   "FFP2 20%", "FFP2 40%", "FFP2 80%")
all_data$Scenario <- factor(all_data$Scenario, levels = desired_order)

# Define custom colors for each scenario
scenario_colors <- c("Baseline"       = "#559e83",  # No Countermeasures (green)
                     "Surgical 20%"   = "#6B95DB",  # Surgical 20% (blue)
                     "Surgical 40%"   = "#ff8b94",  # Surgical 40% (red)
                     "Surgical 80%"   = "#494949",  # Surgical 80% (black/gray)
                     "FFP2 20%"       = "#f4a261",  # FFP2 20% (orange)
                     "FFP2 40%"       = "#e76f51",  # FFP2 40% (coral)
                     "FFP2 80%"       = "#b56576")  # FFP2 80% (pink)

# Create the line plot using ggplot2
png("outside_contagion.png", units = "in", width = 10, height = 6, res = 150)
p <- ggplot(all_data, aes(x = day, y = percentage_infected, color = Scenario)) +
  theme_bw() +
  geom_line() +
  scale_color_manual(values = scenario_colors) +
  labs(title = "",
       x = "Day",
       y = "Percentage Infected") +
  theme(
    legend.position = "bottom",
    legend.key = element_rect(color = "lightgrey", fill = "white"),
    legend.title = element_text(face = "bold", size = 16),
    legend.text = element_text(size = 14),
    legend.key.size = unit(20, "pt"),
    axis.text = element_text(size = 12),
    axis.title = element_text(size = 14, face = "bold")
  )
print(p)
dev.off()
