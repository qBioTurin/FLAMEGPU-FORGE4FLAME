library(dplyr)
library(fdatest)
library(patchwork)
library(ggplot2)

# Function to collect and concatenate data from multiple files
collect_data <- function(mode = "NetLogo",
                         parent_directory = "NetLogo",
                         file_pattern = "result*",
                         directory_pattern = "seed*",
                         filename = "evolution.csv") {
  
  all_data <- data.frame()  # Initialize an empty data frame to store concatenated data
  
  # Check if the parent directory exists
  if (!dir.exists(parent_directory)) {
    message(sprintf("Error: Parent directory '%s' not found.", parent_directory))
    return(all_data)
  }
  
  parent_path <- normalizePath(parent_directory)
  
  if (mode == "NetLogo") {
    message(sprintf("Searching for files in: %s", parent_path))
    file_names <- list.files(parent_path, pattern = file_pattern, full.names = TRUE)
    
    for (file_name in file_names) {
      message(sprintf("Found file: %s", basename(file_name)))
      tryCatch({
        # Read the file
        df <- read.delim(file_name, sep = "\t")

        df <- df %>%
          mutate(time = day,
                 S = susceptible + susceptible.in.quarantine + susceptible.in.quarantine.external.1 + susceptible.in.quarantine.external.2,
                 E = exposed + exposed.in.quarantine + exposed.in.quarantine.external.1 + exposed.in.quarantine.external.2,
                 I = infected + infected.in.quarantine + infected.in.quarantine.external.1 + infected.in.quarantine.external.2,
                 R = removed + removed.in.quarantine + removed.in.quarantine.external.1 + removed.in.quarantine.external.2 + num.immunized) %>%
          dplyr::select(time, S, E, I, R)
        
        
        # Append the data frame to the accumulated data
        all_data <- bind_rows(all_data, df)
      }, error = function(e) {
        message(sprintf("Error reading %s: %s", file_name, e$message))
      })
    }
    
  } else if (mode == "FLAMEGPU2") {
    message(sprintf("Searching for directories in: %s", parent_path))
    directories <- list.dirs(parent_path, recursive = FALSE)
    directories <- directories[grepl(directory_pattern, basename(directories))]
    
    for (directory in directories) {
      message(sprintf("Found directory: %s", basename(directory)))
      
      file_path <- file.path(directory, filename)
      if (file.exists(file_path)) {
        message(sprintf("Found file: %s", file_path))
        tryCatch({
          # Read the file
          df <- read.csv(file_path)
          
          df <- df[which(df$Day < 61),]
          
          df <- df %>%
            mutate(time = Day, S = Susceptible, E = Exposed, I = Infected, R = Recovered) %>%
            dplyr::select(time, S, E, I, R)
          
          
          # Append the data frame to the accumulated data
          all_data <- bind_rows(all_data, df)
        }, error = function(e) {
          message(sprintf("Error reading %s: %s", file_path, e$message))
        })
      } else {
        message(sprintf("File not found: %s", file_path))
      }
    }
    
  } else {
    message(sprintf("Invalid mode: %s. Use 'NetLogo' or 'FLAMEGPU2'.", mode))
  }
  
  return(all_data)
}

# Run fdatest
fdatest_netlogo_vs_flamegpu2 <- function(netlogo_data, flamegpu_data, name){
  traces_df <- data.frame(Day=NULL, value=NULL, compartment=NULL, type=NULL, trace=NULL)
  netlogo_data_S <- netlogo_data_E <- netlogo_data_I <- netlogo_data_R <- data.frame()
  flamegpu_data_S <- flamegpu_data_E <- flamegpu_data_I <- flamegpu_data_R <- data.frame()
  for(i in 0:60) {
    netlogo_data_S_local <-netlogo_data$S[netlogo_data$time == i]
    netlogo_data_E_local <-netlogo_data$E[netlogo_data$time == i]
    netlogo_data_I_local <-netlogo_data$I[netlogo_data$time == i]
    netlogo_data_R_local <-netlogo_data$R[netlogo_data$time == i]
    
    netlogo_data_S <- rbind(netlogo_data_S, netlogo_data_S_local)
    netlogo_data_E <- rbind(netlogo_data_E, netlogo_data_E_local)
    netlogo_data_I <- rbind(netlogo_data_I, netlogo_data_I_local)
    netlogo_data_R <- rbind(netlogo_data_R, netlogo_data_R_local)
    
    traces_df <- rbind(traces_df, data.frame(Day=rep(i, length(netlogo_data_S_local)), value=netlogo_data_S_local, compartment=rep("Susceptible", length(netlogo_data_S_local)), type="NetLogo", trace=seq(1, length(netlogo_data_S_local), 1)))
    traces_df <- rbind(traces_df, data.frame(Day=rep(i, length(netlogo_data_E_local)), value=netlogo_data_E_local, compartment=rep("Exposed", length(netlogo_data_E_local)), type="NetLogo", trace=seq(1, length(netlogo_data_E_local), 1)))
    traces_df <- rbind(traces_df, data.frame(Day=rep(i, length(netlogo_data_I_local)), value=netlogo_data_I_local, compartment=rep("Infected", length(netlogo_data_I_local)), type="NetLogo", trace=seq(1, length(netlogo_data_I_local), 1)))
    traces_df <- rbind(traces_df, data.frame(Day=rep(i, length(netlogo_data_R_local)), value=netlogo_data_R_local, compartment=rep("Recovered", length(netlogo_data_R_local)), type="NetLogo", trace=seq(1, length(netlogo_data_R_local), 1)))
    
    
    flamegpu_data_S_local <-flamegpu_data$S[flamegpu_data$time == i]
    flamegpu_data_E_local <-flamegpu_data$E[flamegpu_data$time == i]
    flamegpu_data_I_local <-flamegpu_data$I[flamegpu_data$time == i]
    flamegpu_data_R_local <-flamegpu_data$R[flamegpu_data$time == i]
    
    flamegpu_data_S <- rbind(flamegpu_data_S, flamegpu_data_S_local)
    flamegpu_data_E <- rbind(flamegpu_data_E, flamegpu_data_E_local)
    flamegpu_data_I <- rbind(flamegpu_data_I, flamegpu_data_I_local)
    flamegpu_data_R <- rbind(flamegpu_data_R, flamegpu_data_R_local)
    
    traces_df <- rbind(traces_df, data.frame(Day=rep(i, length(flamegpu_data_S_local)), value=flamegpu_data_S_local, compartment=rep("Susceptible", length(flamegpu_data_S_local)), type="FLAME GPU 2", trace=seq(1, length(flamegpu_data_S_local), 1)))
    traces_df <- rbind(traces_df, data.frame(Day=rep(i, length(flamegpu_data_E_local)), value=flamegpu_data_E_local, compartment=rep("Exposed", length(flamegpu_data_E_local)), type="FLAME GPU 2", trace=seq(1, length(flamegpu_data_E_local), 1)))
    traces_df <- rbind(traces_df, data.frame(Day=rep(i, length(flamegpu_data_I_local)), value=flamegpu_data_I_local, compartment=rep("Infected", length(flamegpu_data_I_local)), type="FLAME GPU 2", trace=seq(1, length(flamegpu_data_I_local), 1)))
    traces_df <- rbind(traces_df, data.frame(Day=rep(i, length(flamegpu_data_R_local)), value=flamegpu_data_R_local, compartment=rep("Recovered", length(flamegpu_data_R_local)), type="FLAME GPU 2", trace=seq(1, length(flamegpu_data_R_local), 1)))
  }
  
  netlogo_data_S <- t(netlogo_data_S)
  netlogo_data_E <- t(netlogo_data_E)
  netlogo_data_I <- t(netlogo_data_I)
  netlogo_data_R <- t(netlogo_data_R)
  
  flamegpu_data_S <- t(flamegpu_data_S)
  flamegpu_data_E <- t(flamegpu_data_E)
  flamegpu_data_I <- t(flamegpu_data_I)
  flamegpu_data_R <- t(flamegpu_data_R)
  
  ITP.result_S <- ITP2bspline(netlogo_data_S, flamegpu_data_S)
  ITP.result_E <- ITP2bspline(netlogo_data_E, flamegpu_data_E)
  ITP.result_I <- ITP2bspline(netlogo_data_I, flamegpu_data_I)
  ITP.result_R <- ITP2bspline(netlogo_data_R, flamegpu_data_R)
  
  ITP.result_corrected_pval_df <- data.frame(Day=NULL, p_value=NULL, compartment=NULL)
  ITP.result_corrected_pval_df <- rbind(ITP.result_corrected_pval_df, data.frame(Day=seq(0, 60, 1), p_value=ITP.result_S$corrected.pval, compartment="Susceptible"))
  ITP.result_corrected_pval_df <- rbind(ITP.result_corrected_pval_df, data.frame(Day=seq(0, 60, 1), p_value=ITP.result_E$corrected.pval, compartment="Exposed"))
  ITP.result_corrected_pval_df <- rbind(ITP.result_corrected_pval_df, data.frame(Day=seq(0, 60, 1), p_value=ITP.result_I$corrected.pval, compartment="Infected"))
  ITP.result_corrected_pval_df <- rbind(ITP.result_corrected_pval_df, data.frame(Day=seq(0, 60, 1), p_value=ITP.result_R$corrected.pval, compartment="Recovered"))
  
  
  ITP.result_corrected_pval_df$compartment <- factor(ITP.result_corrected_pval_df$compartment, levels=c("Susceptible", "Exposed", "Infected", "Recovered"))
  png(paste0(name, "_p-values.png"), units="in", width=40, height=25, res=150)
  plot_p_values <- ggplot(ITP.result_corrected_pval_df) +
    geom_point(aes(x=Day, y=p_value), size=4) +
    labs(title=paste0(if(name == "NoCountermeasures") "a) " else "c) ", "Adjusted p-values (", gsub("NoC", "No c", name), ")"), x="Day", y="P-value") +
    ylim(0, 1) +
    theme_bw() +
    facet_wrap(~compartment, ncol = 2) +
    theme(axis.title = element_text(size=40), axis.text = element_text(size=35), strip.text = element_text(size=25), plot.title = element_text(size=40, face = "bold"))
  print(plot_p_values)
  dev.off()
  
  traces_df$compartment <- factor(traces_df$compartment, levels=c("Susceptible", "Exposed", "Infected", "Recovered"))
  png(paste0(name, "_traces.png"), units="in", width=40, height=25, res=150)
  plot_traces <- ggplot(traces_df) +
    geom_line(aes(x=Day, y=value, color=type, group=interaction(trace, type)), linetype="dotted", linewidth=2.5) +
    labs(title=paste0(if(name == "NoCountermeasures") "b) " else "d) ", "Simulation traces (", gsub("NoC", "No c", name), ")"), x="Day", y="Population", color="Tool") +
    theme_bw() +
    facet_wrap(~compartment, ncol = 2) +
    scale_color_manual(values = c("#ff8b94", "#494949")) +
    theme(legend.position = "bottom", legend.title = element_text(size=40, face = "bold"), legend.text = element_text(size=40), strip.text = element_text(size=25), legend.key.size = unit(3, 'cm'), axis.title = element_text(size=40, face = "bold"), axis.text = element_text(size=35), plot.title = element_text(size=40, face = "bold"))
  print(plot_traces)
  dev.off()
  
  plot <- plot_p_values / plot_traces
  png(paste0(name, ".png"), units="in", width=40, height=30, res=150)
  print(plot)
  dev.off()
  
  return(list(plot_p_values, plot_traces))
}

netlogo_data <- collect_data(
  mode = "NetLogo",
  parent_directory = "../../NetLogoSchoolModel/ResultsCountermeasures/",
  file_pattern = "result*"
)

flamegpu_data <- collect_data(
  mode = "FLAMEGPU2",
  parent_directory = "../../results/SchoolComparisonCountermeasures",
  directory_pattern = "seed*",
  filename = "evolution.csv"
)

plots <- fdatest_netlogo_vs_flamegpu2(netlogo_data, flamegpu_data, "ComparisonCountermeasures")
plot_p_values_Countermeasures <- plots[[1]]
plot_traces_Countermeasures <- plots[[2]]

netlogo_data <- collect_data(
  mode = "NetLogo",
  parent_directory = "../../NetLogoSchoolModel/ResultsNoCountermeasures/",
  file_pattern = "result*"
)

flamegpu_data <- collect_data(
  mode = "FLAMEGPU2",
  parent_directory = "../../results/SchoolComparisonNoCountermeasures",
  directory_pattern = "seed*",
  filename = "evolution.csv"
)

plots <- fdatest_netlogo_vs_flamegpu2(netlogo_data, flamegpu_data, "ComparisonNoCountermeasures")
plot_p_values_NoCountermeasures <- plots[[1]]
plot_traces_NoCountermeasures <- plots[[2]]




plot <- (plot_p_values_NoCountermeasures + plot_traces_NoCountermeasures) / (plot_p_values_Countermeasures + plot_traces_Countermeasures) +
  plot_layout(guides = "collect", axis_titles = "collect") &
  theme(legend.position = "bottom", legend.box = "vertical")
png("comparison.png", units="in", width=40, height=30, res=300)
print(plot)
dev.off()