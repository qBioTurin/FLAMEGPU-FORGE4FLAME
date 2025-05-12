#ifndef _HOST_FUNCTIONS_CUH_
#define _HOST_FUNCTIONS_CUH_

// #include <fcntl.h>
// #include <unistd.h>
#include <vector>
#include <algorithm>
#include "defines.h"
#include "pugixml.hpp"

using namespace flamegpu;
using namespace std;
using namespace pugi;

namespace host_functions {
    /** 
     * Compare floating points.
    */
    bool compare_float(const double a, const double b, const double epsilon) {
        return fabs(a - b) < epsilon;
    }

    /** 
     * Generate a random number using the given RNG, distribution and parameters for host.
    */
    float cuda_host_rng(HostAPI* FLAMEGPU, unsigned short distribution_id, int type, float a, float b, bool flow_time) {
        float random = (type == TRUNCATED_POSITIVE_NORMAL) ? normal_distr(host_rng[FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX)]): uniform_distr(host_rng[FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX)]);

        if(type == EXPONENTIAL && compare_float((double) random, 1.0f, 1e-10f)){
            do{
                random = uniform_distr(host_rng[FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX)]);
            }while(compare_float((double) random, 1.0f, 1e-10f));
        }
        const float event_time_random = DISTRIBUTION(type, random, a, b);

        //printf("[RANDOM_HOST],%d,%d,%d,%d,%d,%d,%d,%f,%s\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), distribution_id, 0, type, a, b, (flow_time && event_time_random < 1.0f) ? 1.0f: event_time_random, flow_time ? "true" : "false");

        return (flow_time && event_time_random < 1.0f) ? 1.0f: event_time_random;
    }

    /**
     * Helper function to parse comma-separated values.
     */
    vector<float> parseArray(const string& arrayString) {
        vector<float> values;
        stringstream ss(arrayString);
        string item;

        while (getline(ss, item, ',')) {
            values.push_back(stof(item));
        }

        return values;
    }



    /** 
     * Initialization function.
    */
    FLAMEGPU_INIT_FUNCTION(initFunction){        
#ifdef DEBUG
        printf("5,%d,%d,Beginning initFunction for host\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter());
#endif
        unsigned short day = FLAMEGPU->environment.getProperty<unsigned short>(DAY);
        unsigned short week_day = FLAMEGPU->environment.getProperty<unsigned short>(WEEK_DAY);

        host_rng[FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX)].seed(FLAMEGPU->environment.getProperty<unsigned int>(SEED));

        string filename;

        if(day == 1){
            auto num_seird = FLAMEGPU->environment.getMacroProperty<unsigned int, DISEASE_STATES>(COMPARTMENTAL_MODEL);

            filename = "results/" + string(EXPERIMENT_NAME) + "/seed" + to_string(FLAMEGPU->environment.getProperty<unsigned int>(SEED)) + "/counters.csv";
            ofstream counters_file(filename.c_str(), ofstream::out);
            counters_file << "Day,COUNTERS_CREATED_AGENTS_WITH_RATE,COUNTERS_KILLED_AGENTS_WITH_RATE,AGENTS_IN_QUARANTINE,SWABS,NUM_INFECTED_OUTSIDE" << endl;
            counters_file << FLAMEGPU->environment.getProperty<unsigned short>(DAY)-1 << ",0,0,0" << endl;
            counters_file.close();
        }
        else{
            day++;
            FLAMEGPU->environment.setProperty<unsigned short>(DAY, day);
            FLAMEGPU->environment.setProperty<unsigned short>(WEEK_DAY, (week_day + 1) % DAYS_IN_A_WEEK);

            filename = "results/" + string(EXPERIMENT_NAME) + "/seed" + to_string(FLAMEGPU->environment.getProperty<unsigned int>(SEED)) + "/host_rng_state.txt";
            ifstream rng_state(filename.c_str(), ifstream::in);
            rng_state >> host_rng[FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX)];
            rng_state.close();
        }
        
        fflush(stdout);

        printf("4,%d,%d,Simulating day %d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), day);

        
#ifdef DEBUG
        printf("5,%d,%d,Ending initFunction for host\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter());
#endif
    }

    /** 
     * Import macro properties from XML files.
    */
    FLAMEGPU_INIT_FUNCTION(macroPropertyIO) {
#ifdef DEBUG
        printf("5,%d,%d,Beginning macroPropertyIO for host\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter());
#endif
        // Import a macro properties
        FLAMEGPU->environment.importMacroProperty(COORD2INDEX, string("resources/macro_environment/") + COORD2INDEX + ".xml");
        FLAMEGPU->environment.importMacroProperty(ADJMATRIX, string("resources/macro_environment/") + ADJMATRIX + ".xml");
        FLAMEGPU->environment.importMacroProperty(ENV_FLOW, string("resources/macro_environment/") + ENV_FLOW + ".xml");
        FLAMEGPU->environment.importMacroProperty(ENV_FLOW_AREA, string("resources/macro_environment/") + ENV_FLOW_AREA + ".xml");
        FLAMEGPU->environment.importMacroProperty(ENV_FLOW_AGENTLINKED, string("resources/macro_environment/") + ENV_FLOW_AGENTLINKED + ".xml");
        FLAMEGPU->environment.importMacroProperty(ENV_FLOW_DISTR, string("resources/macro_environment/") + ENV_FLOW_DISTR + ".xml");
        FLAMEGPU->environment.importMacroProperty(ENV_FLOW_DISTR_FIRSTPARAM, string("resources/macro_environment/") + ENV_FLOW_DISTR_FIRSTPARAM + ".xml");
        FLAMEGPU->environment.importMacroProperty(ENV_FLOW_DISTR_SECONDPARAM, string("resources/macro_environment/") + ENV_FLOW_DISTR_SECONDPARAM + ".xml");
        FLAMEGPU->environment.importMacroProperty(ENV_ACTIVITY_TYPE, string("resources/macro_environment/") + ENV_ACTIVITY_TYPE + ".xml");
        FLAMEGPU->environment.importMacroProperty(ENV_HOURS_SCHEDULE, string("resources/macro_environment/") + ENV_HOURS_SCHEDULE + ".xml");
        FLAMEGPU->environment.importMacroProperty(ENV_BIRTH_RATE_DISTR, string("resources/macro_environment/") + ENV_BIRTH_RATE_DISTR + ".xml");
        FLAMEGPU->environment.importMacroProperty(ENV_BIRTH_RATE_DISTR_FIRSTPARAM, string("resources/macro_environment/") + ENV_BIRTH_RATE_DISTR_FIRSTPARAM + ".xml");
        FLAMEGPU->environment.importMacroProperty(ENV_BIRTH_RATE_DISTR_SECONDPARAM, string("resources/macro_environment/") + ENV_BIRTH_RATE_DISTR_SECONDPARAM + ".xml");
        FLAMEGPU->environment.importMacroProperty(ENV_EVENTS, string("resources/macro_environment/") + ENV_EVENTS + ".xml");
        FLAMEGPU->environment.importMacroProperty(ENV_EVENTS_AREA, string("resources/macro_environment/") + ENV_EVENTS_AREA + ".xml");
        FLAMEGPU->environment.importMacroProperty(ENV_EVENTS_PROBABILITY, string("resources/macro_environment/") + ENV_EVENTS_PROBABILITY + ".xml");
        FLAMEGPU->environment.importMacroProperty(ENV_EVENTS_AGENTLINKED, string("resources/macro_environment/") + ENV_EVENTS_AGENTLINKED + ".xml");
        FLAMEGPU->environment.importMacroProperty(ENV_EVENTS_STARTTIME, string("resources/macro_environment/") + ENV_EVENTS_STARTTIME + ".xml");
        FLAMEGPU->environment.importMacroProperty(ENV_EVENTS_ENDTIME, string("resources/macro_environment/") + ENV_EVENTS_ENDTIME + ".xml");
        FLAMEGPU->environment.importMacroProperty(ENV_EVENTS_ACTIVITY_TYPE, string("resources/macro_environment/") + ENV_EVENTS_ACTIVITY_TYPE + ".xml");
        FLAMEGPU->environment.importMacroProperty(ENV_EVENTS_DISTR, string("resources/macro_environment/") + ENV_EVENTS_DISTR + ".xml");
        FLAMEGPU->environment.importMacroProperty(ENV_EVENTS_DISTR_FIRSTPARAM, string("resources/macro_environment/") + ENV_EVENTS_DISTR_FIRSTPARAM + ".xml");
        FLAMEGPU->environment.importMacroProperty(ENV_EVENTS_DISTR_SECONDPARAM, string("resources/macro_environment/") + ENV_EVENTS_DISTR_SECONDPARAM + ".xml");
        FLAMEGPU->environment.importMacroProperty(ENV_MASK_TYPE, string("resources/macro_environment/") + ENV_MASK_TYPE + ".xml");
        FLAMEGPU->environment.importMacroProperty(ENV_VENTILATION, string("resources/macro_environment/") + ENV_VENTILATION + ".xml");
        FLAMEGPU->environment.importMacroProperty(ENV_MASK_FRACTION, string("resources/macro_environment/") + ENV_MASK_FRACTION + ".xml");
        FLAMEGPU->environment.importMacroProperty(ENV_VACCINATION_FRACTION, string("resources/macro_environment/") + ENV_VACCINATION_FRACTION + ".xml");
        FLAMEGPU->environment.importMacroProperty(ENV_VACCINATION_EFFICACY, string("resources/macro_environment/") + ENV_VACCINATION_EFFICACY + ".xml");
        FLAMEGPU->environment.importMacroProperty(ENV_VACCINATION_END_OF_IMMUNIZATION_DISTR, string("resources/macro_environment/") + ENV_VACCINATION_END_OF_IMMUNIZATION_DISTR + ".xml");
        FLAMEGPU->environment.importMacroProperty(ENV_VACCINATION_END_OF_IMMUNIZATION_DISTR_FIRSTPARAM, string("resources/macro_environment/") + ENV_VACCINATION_END_OF_IMMUNIZATION_DISTR_FIRSTPARAM + ".xml");
        FLAMEGPU->environment.importMacroProperty(ENV_VACCINATION_END_OF_IMMUNIZATION_DISTR_SECONDPARAM, string("resources/macro_environment/") + ENV_VACCINATION_END_OF_IMMUNIZATION_DISTR_SECONDPARAM + ".xml");
        FLAMEGPU->environment.importMacroProperty(ENV_SWAB_SENSITIVITY, string("resources/macro_environment/") + ENV_SWAB_SENSITIVITY + ".xml");
        FLAMEGPU->environment.importMacroProperty(ENV_SWAB_SPECIFICITY, string("resources/macro_environment/") + ENV_SWAB_SPECIFICITY + ".xml");
        FLAMEGPU->environment.importMacroProperty(ENV_SWAB_DISTR, string("resources/macro_environment/") + ENV_SWAB_DISTR + ".xml");
        FLAMEGPU->environment.importMacroProperty(ENV_SWAB_DISTR_FIRSTPARAM, string("resources/macro_environment/") + ENV_SWAB_DISTR_FIRSTPARAM + ".xml");
        FLAMEGPU->environment.importMacroProperty(ENV_SWAB_DISTR_SECONDPARAM, string("resources/macro_environment/") + ENV_SWAB_DISTR_SECONDPARAM + ".xml");
        FLAMEGPU->environment.importMacroProperty(ENV_QUARANTINE_DAYS_DISTR, string("resources/macro_environment/") + ENV_QUARANTINE_DAYS_DISTR + ".xml");
        FLAMEGPU->environment.importMacroProperty(ENV_QUARANTINE_DAYS_DISTR_FIRSTPARAM, string("resources/macro_environment/") + ENV_QUARANTINE_DAYS_DISTR_FIRSTPARAM + ".xml");
        FLAMEGPU->environment.importMacroProperty(ENV_QUARANTINE_DAYS_DISTR_SECONDPARAM, string("resources/macro_environment/") + ENV_QUARANTINE_DAYS_DISTR_SECONDPARAM + ".xml");
        FLAMEGPU->environment.importMacroProperty(ENV_QUARANTINE_SWAB_SENSITIVITY, string("resources/macro_environment/") + ENV_QUARANTINE_SWAB_SENSITIVITY + ".xml");
        FLAMEGPU->environment.importMacroProperty(ENV_QUARANTINE_SWAB_SPECIFICITY, string("resources/macro_environment/") + ENV_QUARANTINE_SWAB_SPECIFICITY + ".xml");
        FLAMEGPU->environment.importMacroProperty(ENV_QUARANTINE_SWAB_DAYS_DISTR, string("resources/macro_environment/") + ENV_QUARANTINE_SWAB_DAYS_DISTR + ".xml");
        FLAMEGPU->environment.importMacroProperty(ENV_QUARANTINE_SWAB_DAYS_DISTR_FIRSTPARAM, string("resources/macro_environment/") + ENV_QUARANTINE_SWAB_DAYS_DISTR_FIRSTPARAM + ".xml");
        FLAMEGPU->environment.importMacroProperty(ENV_QUARANTINE_SWAB_DAYS_DISTR_SECONDPARAM, string("resources/macro_environment/") + ENV_QUARANTINE_SWAB_DAYS_DISTR_SECONDPARAM + ".xml");
        FLAMEGPU->environment.importMacroProperty(ENV_ROOM_FOR_QUARANTINE_TYPE, string("resources/macro_environment/") + ENV_ROOM_FOR_QUARANTINE_TYPE + ".xml");
        FLAMEGPU->environment.importMacroProperty(ENV_ROOM_FOR_QUARANTINE_AREA, string("resources/macro_environment/") + ENV_ROOM_FOR_QUARANTINE_AREA + ".xml");
        FLAMEGPU->environment.importMacroProperty(ENV_EXTERNAL_SCREENING_FIRST, string("resources/macro_environment/") + ENV_EXTERNAL_SCREENING_FIRST + ".xml");
        FLAMEGPU->environment.importMacroProperty(ENV_EXTERNAL_SCREENING_SECOND, string("resources/macro_environment/") + ENV_EXTERNAL_SCREENING_SECOND + ".xml");
        FLAMEGPU->environment.importMacroProperty(INITIAL_INFECTED, string("resources/macro_environment/") + INITIAL_INFECTED + ".xml");
        FLAMEGPU->environment.importMacroProperty(NUMBER_OF_AGENTS_BY_TYPE, string("resources/macro_environment/") + NUMBER_OF_AGENTS_BY_TYPE + ".xml");
        FLAMEGPU->environment.importMacroProperty(COMPARTMENTAL_MODEL, string("resources/macro_environment/") + COMPARTMENTAL_MODEL + ".xml");
        FLAMEGPU->environment.importMacroProperty(ROOMS_QUANTA_CONCENTRATION, string("resources/macro_environment/") + ROOMS_QUANTA_CONCENTRATION + ".xml");
        FLAMEGPU->environment.importMacroProperty(GLOBAL_RESOURCES, string("resources/macro_environment/") + GLOBAL_RESOURCES + ".xml");
        FLAMEGPU->environment.importMacroProperty(GLOBAL_RESOURCES_COUNTER, string("resources/macro_environment/") + GLOBAL_RESOURCES_COUNTER + ".xml");
        FLAMEGPU->environment.importMacroProperty(SPECIFIC_RESOURCES, string("resources/macro_environment/") + SPECIFIC_RESOURCES + ".xml");
        FLAMEGPU->environment.importMacroProperty(SPECIFIC_RESOURCES_COUNTER, string("resources/macro_environment/") + SPECIFIC_RESOURCES_COUNTER + ".xml"); 
        FLAMEGPU->environment.importMacroProperty(ALTERNATIVE_RESOURCES_TYPE_DET, string("resources/macro_environment/") + ALTERNATIVE_RESOURCES_TYPE_DET + ".xml");
        FLAMEGPU->environment.importMacroProperty(ALTERNATIVE_RESOURCES_AREA_DET, string("resources/macro_environment/") + ALTERNATIVE_RESOURCES_AREA_DET + ".xml");
        FLAMEGPU->environment.importMacroProperty(ALTERNATIVE_RESOURCES_TYPE_RAND, string("resources/macro_environment/") + ALTERNATIVE_RESOURCES_TYPE_RAND + ".xml");
        FLAMEGPU->environment.importMacroProperty(ALTERNATIVE_RESOURCES_AREA_RAND, string("resources/macro_environment/") + ALTERNATIVE_RESOURCES_AREA_RAND + ".xml");
        FLAMEGPU->environment.importMacroProperty(CUDA_RNG_OFFSETS_PEDESTRIAN, string("resources/macro_environment/") + CUDA_RNG_OFFSETS_PEDESTRIAN + ".xml");
        FLAMEGPU->environment.importMacroProperty(CUDA_RNG_OFFSETS_ROOM, string("resources/macro_environment/") + CUDA_RNG_OFFSETS_ROOM + ".xml");

        

#ifdef DEBUG
        printf("5,%d,%d,Ending macroPropertyIO for host\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter());
#endif
    }

    /** 
     * Generate pedestrians with time window as entry type.
    */
    FLAMEGPU_INIT_FUNCTION(generateAgents) {
#ifdef DEBUG
        printf("5,%d,%d,Beginning generate_agents for host\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter());
#endif
        auto intermediate_target_x = FLAMEGPU->environment.getMacroProperty<float, TOTAL_AGENTS_ESTIMATION, SOLUTION_LENGTH>(INTERMEDIATE_TARGET_X);
        auto intermediate_target_y = FLAMEGPU->environment.getMacroProperty<float, TOTAL_AGENTS_ESTIMATION, SOLUTION_LENGTH>(INTERMEDIATE_TARGET_Y);
        auto intermediate_target_z = FLAMEGPU->environment.getMacroProperty<float, TOTAL_AGENTS_ESTIMATION, SOLUTION_LENGTH>(INTERMEDIATE_TARGET_Z);
        auto stay_matrix = FLAMEGPU->environment.getMacroProperty<unsigned int, TOTAL_AGENTS_ESTIMATION, SOLUTION_LENGTH>(STAY);
        auto env_flow = FLAMEGPU->environment.getMacroProperty<int, NUMBER_OF_AGENTS_TYPES, DAYS_IN_A_WEEK, FLOW_LENGTH>(ENV_FLOW);
        auto env_flow_distr = FLAMEGPU->environment.getMacroProperty<int, NUMBER_OF_AGENTS_TYPES, DAYS_IN_A_WEEK, FLOW_LENGTH>(ENV_FLOW_DISTR);
        auto env_flow_distr_firstparam = FLAMEGPU->environment.getMacroProperty<int, NUMBER_OF_AGENTS_TYPES, DAYS_IN_A_WEEK, FLOW_LENGTH>(ENV_FLOW_DISTR_FIRSTPARAM);
        auto env_flow_distr_secondparam = FLAMEGPU->environment.getMacroProperty<int, NUMBER_OF_AGENTS_TYPES, DAYS_IN_A_WEEK, FLOW_LENGTH>(ENV_FLOW_DISTR_SECONDPARAM);
        auto env_hours_schedule = FLAMEGPU->environment.getMacroProperty<int, NUMBER_OF_AGENTS_TYPES, DAYS_IN_A_WEEK, HOURS_SCHEDULE_LENGTH>(ENV_HOURS_SCHEDULE);
        auto env_mask_type = FLAMEGPU->environment.getMacroProperty<int, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_MASK_TYPE);
        auto env_mask_fraction = FLAMEGPU->environment.getMacroProperty<float, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_MASK_FRACTION);
        auto env_vaccination_fraction = FLAMEGPU->environment.getMacroProperty<float, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_VACCINATION_FRACTION);
        auto env_vaccination_efficacy = FLAMEGPU->environment.getMacroProperty<float, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_VACCINATION_EFFICACY);
        auto env_vaccination_end_of_immunization_distr = FLAMEGPU->environment.getMacroProperty<float, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_VACCINATION_END_OF_IMMUNIZATION_DISTR);
        auto env_vaccination_end_of_immunization_distr_firstparam = FLAMEGPU->environment.getMacroProperty<float, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_VACCINATION_END_OF_IMMUNIZATION_DISTR_FIRSTPARAM);
        auto env_vaccination_end_of_immunization_distr_secondparam = FLAMEGPU->environment.getMacroProperty<float, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_VACCINATION_END_OF_IMMUNIZATION_DISTR_SECONDPARAM);
        auto env_swab_distr = FLAMEGPU->environment.getMacroProperty<int, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_SWAB_DISTR);
        auto env_swab_distr_firstparam = FLAMEGPU->environment.getMacroProperty<float, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_SWAB_DISTR_FIRSTPARAM);
        auto env_swab_distr_secondparam = FLAMEGPU->environment.getMacroProperty<float, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_SWAB_DISTR_SECONDPARAM);
        auto num_seird = FLAMEGPU->environment.getMacroProperty<unsigned int, DISEASE_STATES>(COMPARTMENTAL_MODEL);
        auto initial_infected = FLAMEGPU->environment.getMacroProperty<int, NUMBER_OF_AGENTS_TYPES_PLUS_1>(INITIAL_INFECTED);
        auto number_of_agents_by_type = FLAMEGPU->environment.getMacroProperty<int, NUMBER_OF_AGENTS_TYPES_PLUS_1>(NUMBER_OF_AGENTS_BY_TYPE);

        // Generate initial infected agents
        vector<int> selectedIndices;
        vector<int> availableIndices;
        unsigned short init_i = 0;
        unsigned short final_i = NUMBER_OF_AGENTS_TYPES_PLUS_1-1;

        // Handle Global and Specific
        int totalAgents = 0;
        for (unsigned short i = init_i; i < final_i; i++) {
            int numInfected = (int) initial_infected[i];
            int totalAgentsByType = (int) number_of_agents_by_type[i];
            totalAgents += totalAgentsByType;

            // Generate all possible indices for the current type
            vector<int> indices(totalAgentsByType);
            iota(indices.begin(), indices.end(), 0);

            // Shuffle and pick the first `numInfected` indices
            shuffle(indices.begin(), indices.end(), host_rng[FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX)]);
            indices.resize(numInfected);

            // Append selected indices to the result
            selectedIndices.insert(selectedIndices.end(), indices.begin(), indices.end());
        }

        // Handle Random
        // Generate all possible indices, excluding already selected ones
        for (int i = 0; i < totalAgents; i++) {
            if (find(selectedIndices.begin(), selectedIndices.end(), i) != selectedIndices.end()) { // Exclude already selected
                availableIndices.push_back(i);
            }
        }

        // Shuffle and pick `n` random indices
        shuffle(availableIndices.begin(), availableIndices.end(), host_rng[FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX)]);
        availableIndices.resize(initial_infected[NUMBER_OF_AGENTS_TYPES_PLUS_1-1]); // Keep only `n` elements

        // Append new indices to `selectedIndices`
        selectedIndices.insert(selectedIndices.end(), availableIndices.begin(), availableIndices.end());

        xml_document doc_agents;
        doc_agents.load_file("resources/agents_file.xml");

        // Get the root node
        xml_node agents = doc_agents.child("agents");

        // Loop through each xagent node
        for (xml_node xagent = agents.child("xagent"); xagent; xagent = xagent.next_sibling("xagent")) {
            // Extract child nodes of xagent, like <name>, <state>, <x>, etc.
            string name = xagent.child("name").text().as_string();
            short contacts_id = (short) xagent.child("contacts_id").text().as_int();
            int agent_type = xagent.child("agent_type").text().as_int();
            
            const unsigned short week_day = FLAMEGPU->environment.getProperty<unsigned short>(WEEK_DAY);

            HostAgentAPI pedestrian_type = FLAMEGPU->agent(name);
            HostNewAgentAPI new_pedestrian = pedestrian_type.newAgent();

            float x = cuda_host_rng(FLAMEGPU, HOST_OFFSET_X_DISTR_IDX, UNIFORM, FLAMEGPU->environment.getProperty<float, 4>(EXTERN_RANGES, 0), FLAMEGPU->environment.getProperty<float, 4>(EXTERN_RANGES, 1), false);
            float y = YEXTERN;
            float z = cuda_host_rng(FLAMEGPU, HOST_OFFSET_Z_DISTR_IDX, UNIFORM, FLAMEGPU->environment.getProperty<float, 4>(EXTERN_RANGES, 2), FLAMEGPU->environment.getProperty<float, 4>(EXTERN_RANGES, 3), false);

            unsigned short entry_time_index = 0;
            unsigned short empty_days = 0;
            unsigned short weekday_agent = week_day;
            
            bool schedule_found = false;
            while((int) env_flow[agent_type][weekday_agent][entry_time_index] != -1 && !schedule_found){
                if((int) env_hours_schedule[agent_type][weekday_agent][entry_time_index] < START_STEP_TIME)
                    entry_time_index++;
                else
                    schedule_found = true;
            }

            if(!schedule_found)
                entry_time_index = 0;

            while((int) env_flow[agent_type][weekday_agent][entry_time_index] == -1){
                empty_days++;
                weekday_agent = (weekday_agent + 1) % DAYS_IN_A_WEEK;
            }

            int new_agent_state = SUSCEPTIBLE;
            
            float random = cuda_host_rng(FLAMEGPU, HOST_UNIFORM_0_1_DISTR_IDX, UNIFORM, 0.0f, 1.0f, false);
            float random_efficacy = cuda_host_rng(FLAMEGPU, HOST_UNIFORM_0_1_DISTR_IDX, UNIFORM, 0.0f, 1.0f, false);
            unsigned short vaccination_end_of_immunization_days = 0;
            if(random < (float) env_vaccination_fraction[0][agent_type] && random_efficacy < (float) env_vaccination_efficacy[0][agent_type]){
                new_agent_state = RECOVERED;
#ifdef REINFECTION
                vaccination_end_of_immunization_days = (unsigned short) max(0.0f, round(cuda_host_rng(FLAMEGPU, HOST_VACCINATION_END_OF_IMMUNIZATION_DISTR_IDX, (int) env_vaccination_end_of_immunization_distr[0][agent_type], (float) env_vaccination_end_of_immunization_distr_firstparam[0][agent_type], (float) env_vaccination_end_of_immunization_distr_secondparam[0][agent_type], false)));
#endif
            }

            unsigned short infection_days = 0;
            unsigned short fatality_days = 0;
            if(new_agent_state == SUSCEPTIBLE && find(selectedIndices.begin(), selectedIndices.end(), contacts_id) != selectedIndices.end()){
                new_agent_state = INFECTED;
                infection_days = (unsigned short) max(0.0f, round(cuda_host_rng(FLAMEGPU, HOST_INFECTION_DISTR_IDX, FLAMEGPU->environment.getProperty<unsigned short, 3>(MEAN_INFECTION_DAYS, 0), (float) FLAMEGPU->environment.getProperty<unsigned short, 3>(MEAN_INFECTION_DAYS, 1), (float) FLAMEGPU->environment.getProperty<unsigned short, 3>(MEAN_INFECTION_DAYS, 2), false)));
#ifdef FATALITY
                fatality_days = (unsigned short) max(0.0f, round(cuda_host_rng(FLAMEGPU, HOST_FATALITY_DISTR_IDX, FLAMEGPU->environment.getProperty<unsigned short, 3>(MEAN_FATALITY_DAYS, 0), (float) FLAMEGPU->environment.getProperty<unsigned short, 3>(MEAN_FATALITY_DAYS, 1), (float) FLAMEGPU->environment.getProperty<unsigned short, 3>(MEAN_FATALITY_DAYS, 2), false)));
#endif
            }

            new_pedestrian.setVariable<float>(X, x);
            new_pedestrian.setVariable<float>(Y, INVISIBLE_AGENT_Y);
            new_pedestrian.setVariable<float>(Z, z);
            new_pedestrian.setVariable<float, 3>(FINAL_TARGET, {x, y, z});
            new_pedestrian.setVariable<int>(DISEASE_STATE, new_agent_state);
            new_pedestrian.setVariable<short>(CONTACTS_ID, contacts_id);
            new_pedestrian.setVariable<int>(AGENT_TYPE, agent_type);
            new_pedestrian.setVariable<int>(MASK_TYPE, (cuda_host_rng(FLAMEGPU, HOST_UNIFORM_0_1_DISTR_IDX, UNIFORM, 0.0f, 1.0f, false) < (float) env_mask_fraction[0][agent_type]) ? (int) env_mask_type[0][agent_type]: NO_MASK);
            new_pedestrian.setVariable<unsigned short>(END_OF_IMMUNIZATION_DAYS, vaccination_end_of_immunization_days);
            new_pedestrian.setVariable<unsigned short>(INFECTION_DAYS, infection_days);
            new_pedestrian.setVariable<unsigned short>(FATALITY_DAYS, fatality_days);
            new_pedestrian.setVariable<unsigned short>(WEEK_DAY_FLOW, weekday_agent);

            int swab_steps = -1;
            if((int) env_swab_distr[0][agent_type] != NO_SWAB)
                swab_steps = round(cuda_host_rng(FLAMEGPU, HOST_SWAB_DISTR_IDX, (int) env_swab_distr[0][agent_type], (float) (STEPS_IN_A_DAY * env_swab_distr_firstparam[0][agent_type]), (float) (STEPS_IN_A_DAY * env_swab_distr_secondparam[0][agent_type]), true));
            new_pedestrian.setVariable<int>(SWAB_STEPS, swab_steps);

            const unsigned short initial_stay = empty_days * STEPS_IN_A_DAY + ((int) env_hours_schedule[agent_type][weekday_agent][entry_time_index]) - START_STEP_TIME + cuda_host_rng(FLAMEGPU, HOST_FLOW_DISTR_IDX, (int) env_flow_distr[agent_type][weekday_agent][entry_time_index], (float) env_flow_distr_firstparam[agent_type][weekday_agent][entry_time_index], (float) env_flow_distr_secondparam[agent_type][weekday_agent][entry_time_index], true);
            stay_matrix[contacts_id][0] = initial_stay;

            intermediate_target_x[contacts_id][0] = x;
            intermediate_target_y[contacts_id][0] = y;
            intermediate_target_z[contacts_id][0] = z;
            new_pedestrian.setVariable<int>(WAITING_ROOM_TIME, 0);
            new_pedestrian.setVariable<int>(WAITING_ROOM_FLAG, 0);
            new_pedestrian.setVariable<int>(ENTRY_EXIT_FLAG, STAYING_IN_WAITING_ROOM);
            new_pedestrian.setVariable<short>(NODE_WAITING_FOR, -1);
            new_pedestrian.setVariable<short>(ACTUAL_EVENT_NODE, -1);

            num_seird[new_agent_state]++;
        }

        xml_document doc;
        doc.load_file("resources/rooms_file.xml");

        // Get the root node
        xml_node rooms = doc.child("rooms");

        // Loop through each xagent node
        for (xml_node xagent = rooms.child("xagent"); xagent; xagent = xagent.next_sibling("xagent")) {
            // Extract child nodes of xagent, like <name>, <state>, <x>, etc.
            string name = xagent.child("name").text().as_string();
            float x = xagent.child("x").text().as_float();
            float y = xagent.child("y").text().as_float();
            float z = xagent.child("z").text().as_float();
            float length_obj = xagent.child("length_obj").text().as_float();
            float width_obj = xagent.child("width_obj").text().as_float();
            float height_obj = xagent.child("height_obj").text().as_float();
            float yaw = xagent.child("yaw").text().as_float();
            unsigned char init_room = (unsigned char) xagent.child("init_room").text().as_int();
            int area = xagent.child("area").text().as_int();
            int color_id = xagent.child("color_id").text().as_int();
            int type = xagent.child("type").text().as_int();
            float volume = xagent.child("volume").text().as_float();
            float room_quanta_concentration = xagent.child("room_quanta_concentration").text().as_float();
            unsigned short x_center = (unsigned short) xagent.child("x_center").text().as_int();
            unsigned short y_center = (unsigned short) xagent.child("y_center").text().as_int();
            unsigned short z_center = (unsigned short) xagent.child("z_center").text().as_int();

            HostAgentAPI room_type = FLAMEGPU->agent(name);
            HostNewAgentAPI new_room = room_type.newAgent();

            new_room.setVariable<float>(X, x);
            new_room.setVariable<float>(Y, y);
            new_room.setVariable<float>(Z, z);
            new_room.setVariable<float>(LENGTH_OBJ, length_obj);
            new_room.setVariable<float>(WIDTH_OBJ, width_obj);
            new_room.setVariable<float>(HEIGHT_OBJ, height_obj);
            new_room.setVariable<float>(YAW, yaw);
            new_room.setVariable<unsigned char>(INIT_ROOM, init_room);
            new_room.setVariable<int>(AREA, area);
            new_room.setVariable<int>(COLOR_ID, color_id);
            new_room.setVariable<int>(TYPE, type);

            if(name != FILLINGROOM_AGENT_STRING){
                new_room.setVariable<float>(VOLUME, volume);
                new_room.setVariable<float>(ROOM_QUANTA_CONCENTRATION, room_quanta_concentration);
                new_room.setVariable<unsigned short>(X_CENTER, x_center);
                new_room.setVariable<unsigned short>(Y_CENTER, y_center);
                new_room.setVariable<unsigned short>(Z_CENTER, z_center);
            }
        }

        string filename = "results/" + string(EXPERIMENT_NAME) + "/seed" + to_string(FLAMEGPU->environment.getProperty<unsigned int>(SEED)) + "/evolution.csv";
        ofstream evolution_file(filename.c_str(), ofstream::out);
        evolution_file << "Day,Susceptible,Exposed,Infected,Recovered,Died" << endl;

        const unsigned short day = FLAMEGPU->environment.getProperty<unsigned short>(DAY);

        evolution_file << day-1 << ",";
        for(int i = 0; i < DISEASE_STATES; i++){
            if(i == (DISEASE_STATES - 1)){
                evolution_file << num_seird[i];
            }
            else{
                evolution_file << num_seird[i] << ",";
            }
        }

        evolution_file << endl;
        evolution_file.close();

#ifdef DEBUG
        printf("5,%d,%d,Ending generate_agents for host\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter());
#endif
    }

    /** 
     * Update the simulation day and log.
    */
    FLAMEGPU_STEP_FUNCTION(updateDayAndLog) {
#ifdef DEBUG
        printf("5,%d,%d,Beginning updateDayAndLog for host\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter());
#endif
        if(FLAMEGPU->getStepCounter() && !((FLAMEGPU->getStepCounter() + START_STEP_TIME) % STEPS_IN_A_HOUR)){
            auto contacts_matrix = FLAMEGPU->environment.getMacroProperty<unsigned int, NUMBER_OF_AGENTS_TYPES_PLUS_1, NUMBER_OF_AGENTS_TYPES_PLUS_1>(CONTACTS_MATRIX);

            for(int i=0;i<NUMBER_OF_AGENTS_TYPES;i++){
                for(int j=0;j<=i;j++){
                    if((unsigned int) contacts_matrix[i][j] != 0){
                        printf("2,%d,%d,%d,%d,%d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), i, j, (unsigned int) contacts_matrix[i][j]);
                        contacts_matrix[i][j] = 0;
                    }
                }
            }
        }

        if(FLAMEGPU->getStepCounter() && !((FLAMEGPU->getStepCounter() + START_STEP_TIME) % STEPS_IN_A_DAY)){
            unsigned short day = FLAMEGPU->environment.getProperty<unsigned short>(DAY) + 1;
            //printf("DAY preso giusto? %d\n", day);
            unsigned short week_day = (FLAMEGPU->environment.getProperty<unsigned short>(WEEK_DAY) + 1) % DAYS_IN_A_WEEK;
            

            FLAMEGPU->environment.setProperty<unsigned short>(DAY, day);
            FLAMEGPU->environment.setProperty<unsigned short>(WEEK_DAY, week_day);
            printf("4,%d,%d,Simulating day %d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), day);
           
            auto num_seird = FLAMEGPU->environment.getMacroProperty<unsigned int, DISEASE_STATES>(COMPARTMENTAL_MODEL);
            auto counters = FLAMEGPU->environment.getMacroProperty<unsigned int, NUM_COUNTERS>(COUNTERS);

            string filename = "results/" + string(EXPERIMENT_NAME) + "/seed" + to_string(FLAMEGPU->environment.getProperty<unsigned int>(SEED)) + "/evolution.csv";
            ofstream evolution_file(filename.c_str(), ofstream::app);
            evolution_file << day-1 << ",";
            for(int i = 0; i < DISEASE_STATES; i++){
                if(i == (DISEASE_STATES - 1)){
                    evolution_file << num_seird[i];
                }
                else{
                    evolution_file << num_seird[i] << ",";
                }
            }

            evolution_file << endl;
            evolution_file.close();

            filename = "results/" + string(EXPERIMENT_NAME) + "/seed" + to_string(FLAMEGPU->environment.getProperty<unsigned int>(SEED)) + "/counters.csv";
            ofstream counters_file(filename.c_str(), ofstream::app);
            counters_file << day-1 << ",";
            for(int i = 0; i < NUM_COUNTERS; i++){
                if(i == (NUM_COUNTERS - 1)){
                    counters_file << counters[i];
                }
                else{
                    counters_file << counters[i] << ",";
                }
            }

            counters_file << endl;
            counters_file.close();

            #ifdef DEBUG

            auto global_resources_counter = FLAMEGPU->environment.getMacroProperty<unsigned int, V>(GLOBAL_RESOURCES_COUNTER);
            auto specific_resources_counter = FLAMEGPU->environment.getMacroProperty<unsigned int, NUMBER_OF_AGENTS_TYPES, V>(SPECIFIC_RESOURCES_COUNTER);
            printf("[DEBUG],%d,%d,GLOBAL_RESOURCES_COUNTER: ", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter());
            for(int i = 0; i < V; i++){
                printf("%d,", (unsigned int) global_resources_counter[i]);
            }
            printf("\n");
            printf("[DEBUG],%d,%d,SPECIFIC_RESOURCES_COUNTER: ", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter());
            for(int i = 0; i < NUMBER_OF_AGENTS_TYPES; i++){
                for(int j = 0; j < V; j++){
                    printf("%d, ", (unsigned int) specific_resources_counter[i][j]);
                }
                printf("\n");
            }
            printf("\n");
            
            #endif


         }

#ifdef DEBUG
        printf("5,%d,%d,Ending updateDayAndLog for host\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter());
#endif
    }

    /** 
     * Generate pedestrians with daily rate as entry type.
    */
    FLAMEGPU_STEP_FUNCTION(birth) {
        if(!((FLAMEGPU->getStepCounter() + START_STEP_TIME) % STEPS_IN_A_DAY)){
#ifdef DEBUG
            printf("5,%d,%d,Beginning birth for host\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter());
#endif
            const unsigned short day = FLAMEGPU->environment.getProperty<unsigned short>(DAY);
            const unsigned short week_day = FLAMEGPU->environment.getProperty<unsigned short>(WEEK_DAY);
            
            short contacts_id = FLAMEGPU->environment.getProperty<short>(NEXT_CONTACTS_ID);

            auto intermediate_target_x = FLAMEGPU->environment.getMacroProperty<float, TOTAL_AGENTS_ESTIMATION, SOLUTION_LENGTH>(INTERMEDIATE_TARGET_X);
            auto intermediate_target_y = FLAMEGPU->environment.getMacroProperty<float, TOTAL_AGENTS_ESTIMATION, SOLUTION_LENGTH>(INTERMEDIATE_TARGET_Y);
            auto intermediate_target_z = FLAMEGPU->environment.getMacroProperty<float, TOTAL_AGENTS_ESTIMATION, SOLUTION_LENGTH>(INTERMEDIATE_TARGET_Z);
            auto stay_matrix = FLAMEGPU->environment.getMacroProperty<unsigned int, TOTAL_AGENTS_ESTIMATION, SOLUTION_LENGTH>(STAY);
            auto num_seird = FLAMEGPU->environment.getMacroProperty<unsigned int, DISEASE_STATES>(COMPARTMENTAL_MODEL);
            auto env_hours_schedule = FLAMEGPU->environment.getMacroProperty<int, NUMBER_OF_AGENTS_TYPES, DAYS_IN_A_WEEK, HOURS_SCHEDULE_LENGTH>(ENV_HOURS_SCHEDULE);
            auto env_rate_distr = FLAMEGPU->environment.getMacroProperty<int, NUMBER_OF_AGENTS_TYPES, DAYS_IN_A_WEEK, HOURS_SCHEDULE_LENGTH>(ENV_BIRTH_RATE_DISTR);
            auto env_rate_distr_firstparam = FLAMEGPU->environment.getMacroProperty<int, NUMBER_OF_AGENTS_TYPES, DAYS_IN_A_WEEK, HOURS_SCHEDULE_LENGTH>(ENV_BIRTH_RATE_DISTR_FIRSTPARAM);
            auto env_rate_distr_secondparam = FLAMEGPU->environment.getMacroProperty<int, NUMBER_OF_AGENTS_TYPES, DAYS_IN_A_WEEK, HOURS_SCHEDULE_LENGTH>(ENV_BIRTH_RATE_DISTR_SECONDPARAM);
            auto env_mask_type = FLAMEGPU->environment.getMacroProperty<int, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_MASK_TYPE);
            auto env_mask_fraction = FLAMEGPU->environment.getMacroProperty<float, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_MASK_FRACTION);
            auto env_vaccination_fraction = FLAMEGPU->environment.getMacroProperty<float, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_VACCINATION_FRACTION);
            auto env_vaccination_efficacy = FLAMEGPU->environment.getMacroProperty<float, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_VACCINATION_EFFICACY);
            auto env_vaccination_end_of_immunization_distr = FLAMEGPU->environment.getMacroProperty<float, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_VACCINATION_END_OF_IMMUNIZATION_DISTR);
            auto env_vaccination_end_of_immunization_distr_firstparam = FLAMEGPU->environment.getMacroProperty<float, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_VACCINATION_END_OF_IMMUNIZATION_DISTR_FIRSTPARAM);
            auto env_vaccination_end_of_immunization_distr_secondparam = FLAMEGPU->environment.getMacroProperty<float, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_VACCINATION_END_OF_IMMUNIZATION_DISTR_SECONDPARAM);
            auto env_swab_distr = FLAMEGPU->environment.getMacroProperty<int, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_SWAB_DISTR);
            auto env_swab_distr_firstparam = FLAMEGPU->environment.getMacroProperty<float, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_SWAB_DISTR_FIRSTPARAM);
            auto env_swab_distr_secondparam = FLAMEGPU->environment.getMacroProperty<float, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_SWAB_DISTR_SECONDPARAM);
            auto counters = FLAMEGPU->environment.getMacroProperty<unsigned int, NUM_COUNTERS>(COUNTERS);

            for(int i = NUMBER_OF_AGENTS_TYPES_WITHOUT_A_RATE; i < NUMBER_OF_AGENTS_TYPES; i++){
                unsigned short slot = 0;
                while((int) env_rate_distr[i][week_day][slot] != -1){
                    unsigned short random_agent = (unsigned short) cuda_host_rng(FLAMEGPU, HOST_RATE_DISTR_IDX, (int) env_rate_distr[i][week_day][slot], (float) env_rate_distr_firstparam[i][week_day][slot], (float) env_rate_distr_secondparam[i][week_day][slot], true);

                    for(int j = 0; j < random_agent; j++){
                        int new_agent_state = SUSCEPTIBLE;
                        
                        HostAgentAPI pedestrian = FLAMEGPU->agent("pedestrian");

                        float x = cuda_host_rng(FLAMEGPU, HOST_OFFSET_X_DISTR_IDX, UNIFORM, FLAMEGPU->environment.getProperty<float, 4>(EXTERN_RANGES, 0), FLAMEGPU->environment.getProperty<float, 4>(EXTERN_RANGES, 1), false);
                        float y = YEXTERN;
                        float z = cuda_host_rng(FLAMEGPU, HOST_OFFSET_Z_DISTR_IDX, UNIFORM, FLAMEGPU->environment.getProperty<float, 4>(EXTERN_RANGES, 2), FLAMEGPU->environment.getProperty<float, 4>(EXTERN_RANGES, 3), false);

                        float random = cuda_host_rng(FLAMEGPU, HOST_UNIFORM_0_1_DISTR_IDX, UNIFORM, 0.0f, 1.0f, false);
                        float random_efficacy = cuda_host_rng(FLAMEGPU, HOST_UNIFORM_0_1_DISTR_IDX, UNIFORM, 0.0f, 1.0f, false);
                        unsigned short vaccination_end_of_immunization_days = 0;
                        if(random < (float) env_vaccination_fraction[day-1][i] && random_efficacy < (float) env_vaccination_efficacy[day-1][i]){
                            new_agent_state = RECOVERED;
#ifdef REINFECTION
                            vaccination_end_of_immunization_days = (unsigned short) cuda_host_rng(FLAMEGPU, HOST_VACCINATION_END_OF_IMMUNIZATION_DISTR_IDX, (int) env_vaccination_end_of_immunization_distr[day-1][i], (float) env_vaccination_end_of_immunization_distr_firstparam[day-1][i], (float) env_vaccination_end_of_immunization_distr_secondparam[day-1][i], true);
#endif
                        }

                        HostNewAgentAPI new_pedestrian = pedestrian.newAgent();

                        counters[COUNTERS_CREATED_AGENTS_WITH_RATE]++;

                        new_pedestrian.setVariable<float>(X, x);
                        new_pedestrian.setVariable<float>(Y, INVISIBLE_AGENT_Y);
                        new_pedestrian.setVariable<float>(Z, z);
                        new_pedestrian.setVariable<float, 3>(FINAL_TARGET, {x, y, z});
                        new_pedestrian.setVariable<short>(CONTACTS_ID, contacts_id);
                        new_pedestrian.setVariable<int>(DISEASE_STATE, new_agent_state);
                        new_pedestrian.setVariable<int>(MASK_TYPE, (cuda_host_rng(FLAMEGPU, HOST_UNIFORM_0_1_DISTR_IDX, UNIFORM, 0.0f, 1.0f, false) < (float) env_mask_fraction[day-1][i]) ? (int) env_mask_type[day-1][i]: NO_MASK);
                        new_pedestrian.setVariable<int>(AGENT_TYPE, i);
                        new_pedestrian.setVariable<unsigned short>(END_OF_IMMUNIZATION_DAYS, vaccination_end_of_immunization_days);
                        new_pedestrian.setVariable<unsigned short>(AGENT_WITH_A_RATE, AGENT_WITH_RATE);
                        new_pedestrian.setVariable<unsigned short>(SEVERITY, MINOR);
                        new_pedestrian.setVariable<unsigned short>(IDENTIFIED_INFECTED, NOT_IDENTIFIED);
                        new_pedestrian.setVariable<unsigned short>(WEEK_DAY_FLOW, week_day);

                        int swab_steps = -1;
                        if((int) env_swab_distr[day-1][i] != NO_SWAB)
                            swab_steps = round(cuda_host_rng(FLAMEGPU, HOST_SWAB_DISTR_IDX, (int) env_swab_distr[day-1][i], (float) (STEPS_IN_A_DAY * env_swab_distr_firstparam[day-1][i]), (float) (STEPS_IN_A_DAY * env_swab_distr_secondparam[day-1][i]), true));

                        new_pedestrian.setVariable<int>(SWAB_STEPS, swab_steps);

                        const unsigned short initial_stay = (unsigned short) cuda_host_rng(FLAMEGPU, HOST_HOURS_SCHEDULE_DISTR_IDX, UNIFORM, (float) env_hours_schedule[i][week_day][2 * slot], (float) env_hours_schedule[i][week_day][2 * slot + 1], true);
                        stay_matrix[contacts_id][0] = initial_stay;

                        intermediate_target_x[contacts_id][0] = x;
                        intermediate_target_y[contacts_id][0] = y;
                        intermediate_target_z[contacts_id][0] = z;
                        new_pedestrian.setVariable<short>(NODE_WAITING_FOR, -1);
                        new_pedestrian.setVariable<short>(ACTUAL_EVENT_NODE, -1);

                        contacts_id = contacts_id + 1;
                        num_seird[new_agent_state]++;
                    }

                    slot++;
                }
            }

            FLAMEGPU->environment.setProperty<short>(NEXT_CONTACTS_ID, contacts_id);
#ifdef DEBUG
            printf("5,%d,%d,Ending birth for host\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter());
#endif
        }
    }

    /** 
     * End of simulation condition.
    */
    FLAMEGPU_EXIT_CONDITION(endOfSimulation){
#ifdef DEBUG
        printf("5,%d,%d,Beginning endOfSimulation for host\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter());
#endif
        if(FLAMEGPU->environment.getProperty<unsigned short>(DAY) == DAYS){
#ifdef DEBUG
            printf("5,%d,%d,Ending endOfSimulation for host\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter());
#endif
            return EXIT;
        }

#ifdef DEBUG
        printf("5,%d,%d,Ending endOfSimulation for host\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter());
#endif
        return CONTINUE;
    }

    /** 
     * End the simulation logging some information.
    */
    FLAMEGPU_EXIT_FUNCTION(exitFunction){
#ifdef DEBUG
        printf("5,%d,%d,Beginning exitFunction for host\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter());
#endif
        // Log the environment macro properties
        auto num_seird = FLAMEGPU->environment.getMacroProperty<unsigned int, DISEASE_STATES>(COMPARTMENTAL_MODEL);
        auto counters = FLAMEGPU->environment.getMacroProperty<unsigned int, NUM_COUNTERS>(COUNTERS);

        string filename;

        filename = "results/" + string(EXPERIMENT_NAME) + "/seed" + to_string(FLAMEGPU->environment.getProperty<unsigned int>(SEED)) + "/evolution.csv";
        ofstream evolution_file(filename.c_str(), ofstream::app);
        evolution_file << FLAMEGPU->environment.getProperty<unsigned short>(DAY) << ",";
        for(int i = 0; i < DISEASE_STATES; i++){
            if(i == (DISEASE_STATES - 1)){
                evolution_file << num_seird[i];
            }
            else{
                evolution_file << num_seird[i] << ",";
            }
        }
        evolution_file << endl;
        evolution_file.close();

        filename = "results/" + string(EXPERIMENT_NAME) + "/seed" + to_string(FLAMEGPU->environment.getProperty<unsigned int>(SEED)) + "/counters.csv";
        ofstream counters_file(filename.c_str(), ofstream::app);
        counters_file << FLAMEGPU->environment.getProperty<unsigned short>(DAY) << ",";
        for(int i = 0; i < NUM_COUNTERS; i++){
            if(i == (NUM_COUNTERS - 1)){
                counters_file << counters[i];
            }
            else{
                counters_file << counters[i] << ",";
            }
        }
        counters_file << endl;
        counters_file.close();

        filename = "results/" + string(EXPERIMENT_NAME) + "/seed" + to_string(FLAMEGPU->environment.getProperty<unsigned int>(SEED)) + "/host_rng_state.txt";
        ofstream rng_state(filename.c_str(), ofstream::out);
        rng_state << host_rng[FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX)];
        rng_state.close();

        printf("4,%d,%d,Simulation completed.\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter());
#ifdef DEBUG
        printf("5,%d,%d,Ending exitFunction for host\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter());
#endif
    }
}
#endif //_HOST_FUNCTIONS_CUH_
