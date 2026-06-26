#ifndef _AGENT_FUNCTIONS_CUH_
#define _AGENT_FUNCTIONS_CUH_

#include "defines.h"
#include "device_functions.cuh"

#include <math.h>

using namespace std;
using namespace flamegpu;
using namespace device_functions;


/**
    initCondition

    Execute a function if the agent is enabled
*/
FLAMEGPU_AGENT_FUNCTION_CONDITION(initCondition) {
    return FLAMEGPU->getVariable<unsigned char>(INIT);
}




/**
    CUDAInit

    Condition: -

    CUDA RNGs initialization, outside contagion, and external screening.
*/
FLAMEGPU_AGENT_FUNCTION(CUDAInit, MessageBucket, MessageNone) {
#if defined(DEBUG) && !defined(ENSEMBLE)
    printf("5,%d,%d,Beginning CUDAInit for agent with id %d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID));
#endif
    // CUDA initialization
    if(!FLAMEGPU->getVariable<unsigned short>(CUDA_INITIALIZED)){
        auto cuda_rng_offsets_pedestrian = FLAMEGPU->environment.getMacroProperty<unsigned int, TOTAL_AGENTS_ESTIMATION>(CUDA_RNG_OFFSETS_PEDESTRIAN);

        curand_init((unsigned long long) FLAMEGPU->environment.getProperty<unsigned int>(SEED), FLAMEGPU->getVariable<short>(CONTACTS_ID)+1, (unsigned int) cuda_rng_offsets_pedestrian[FLAMEGPU->getVariable<short>(CONTACTS_ID)], &cuda_pedestrian_states[FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX)][FLAMEGPU->getVariable<short>(CONTACTS_ID)]);
        FLAMEGPU->setVariable<unsigned short>(CUDA_INITIALIZED, 1);
    }

    float agent_pos[3] = {FLAMEGPU->getVariable<float>(X), FLAMEGPU->getVariable<float>(Y), FLAMEGPU->getVariable<float>(Z)};

    FLAMEGPU->setVariable<char>(CAN_MOVE, 0);

    // Save previous position before updating it
    FLAMEGPU->setVariable<float>(X_PREV, FLAMEGPU->getVariable<float>(X));
    FLAMEGPU->setVariable<float>(Y_PREV, FLAMEGPU->getVariable<float>(Y));
    FLAMEGPU->setVariable<float>(Z_PREV, FLAMEGPU->getVariable<float>(Z));

    // If the agent exits from the environment, I mark it for the outside contagion and the external screening
    if(agent_pos[1] == INVISIBLE_AGENT_Y && !FLAMEGPU->getVariable<unsigned char>(EXITED_FROM_ENVIRONMENT))
        FLAMEGPU->setVariable<unsigned char>(EXITED_FROM_ENVIRONMENT, 1);

    if(agent_pos[1] != INVISIBLE_AGENT_Y && FLAMEGPU->getVariable<unsigned char>(EXITED_FROM_ENVIRONMENT) != 0){
         FLAMEGPU->setVariable<unsigned char>(EXITED_FROM_ENVIRONMENT, 0);
    }

    if(FLAMEGPU->getVariable<unsigned char>(EXITED_FROM_ENVIRONMENT) == 1){
        outside_contagion(FLAMEGPU);
        external_screening(FLAMEGPU);

        FLAMEGPU->setVariable<unsigned char>(EXITED_FROM_ENVIRONMENT, 2);
    }

#if defined(DEBUG) && !defined(ENSEMBLE)
    printf("5,%d,%d,Beginning CUDAInit for agent with id %d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID));
#endif
    return ALIVE;
}

FLAMEGPU_AGENT_FUNCTION(CUDAContagionScreening, MessageBucket, MessageNone) {
#if defined(DEBUG) && !defined(ENSEMBLE)
    printf("5,%d,%d,Beginning CUDAContagionScreening for agent with id %d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID));
#endif
    unsigned short target_index = FLAMEGPU->getVariable<unsigned short>(TARGET_INDEX);
    unsigned char disease_state = FLAMEGPU->getVariable<unsigned char>(DISEASE_STATE);

    auto stay_matrix = FLAMEGPU->environment.getMacroProperty<unsigned int, TOTAL_AGENTS_ESTIMATION, SOLUTION_LENGTH>(STAY);

    const short contacts_id = FLAMEGPU->getVariable<short>(CONTACTS_ID);

    // Contagion processes
    flamegpu::AGENT_STATUS state = ALIVE;

    // Contagion processes (contacts and aerosol)
    contagion_processes(FLAMEGPU);

    if(!((FLAMEGPU->getStepCounter() + START_STEP_TIME + 1) % STEPS_IN_A_DAY)){
        // Update disease state
        state = update_infection(FLAMEGPU);

        if(state == DEAD){
            // Set stay to 1 to correctly update the agent death
            stay_matrix[contacts_id][target_index].exchange(1);

            disease_state = DIED;
            FLAMEGPU->setVariable<unsigned char>(DISEASE_STATE, disease_state);

#if defined(DEBUG) && !defined(ENSEMBLE)
            printf("5,%d,%d,Ending CUDAContagionScreening for agent with id %d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID));
#endif
            return ALIVE;
        }
    }

    if(!((FLAMEGPU->getStepCounter() + START_STEP_TIME - 1) % STEPS_IN_A_DAY)){
        // Update daily What-If
        const unsigned short day = FLAMEGPU->environment.getProperty<unsigned short>(DAY);
        const short agent_type = FLAMEGPU->getVariable<short>(AGENT_TYPE);

        // Update mask usage
        auto env_mask_type = FLAMEGPU->environment.getMacroProperty<unsigned short, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_MASK_TYPE);
        auto env_mask_fraction = FLAMEGPU->environment.getMacroProperty<float, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_MASK_FRACTION);

        FLAMEGPU->setVariable<unsigned char>(MASK_TYPE, (cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_UNIFORM_0_1_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX)], UNIFORM, contacts_id, 0.0f, 1.0f, false) < (float) env_mask_fraction[day-1][agent_type]) ? (unsigned short) env_mask_type[day-1][agent_type]: NO_MASK);

        // Update vaccination
        auto env_vaccination_fraction = FLAMEGPU->environment.getMacroProperty<float, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_VACCINATION_FRACTION);
        auto env_vaccination_efficacy = FLAMEGPU->environment.getMacroProperty<float, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_VACCINATION_EFFICACY);
        auto env_vaccination_end_of_immunization_distr = FLAMEGPU->environment.getMacroProperty<float, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_VACCINATION_END_OF_IMMUNIZATION_DISTR);
        auto env_vaccination_end_of_immunization_distr_firstparam = FLAMEGPU->environment.getMacroProperty<float, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_VACCINATION_END_OF_IMMUNIZATION_DISTR_FIRSTPARAM);
        auto env_vaccination_end_of_immunization_distr_secondparam = FLAMEGPU->environment.getMacroProperty<float, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_VACCINATION_END_OF_IMMUNIZATION_DISTR_SECONDPARAM);

        float random = cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_UNIFORM_0_1_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX)], UNIFORM, contacts_id, 0.0f, 1.0f, false);
        float random_efficacy = cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_UNIFORM_0_1_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX)], UNIFORM, contacts_id, 0.0f, 1.0f, false);

        if(disease_state == SUSCEPTIBLE && random < ((float) env_vaccination_fraction[day-1][agent_type]) && random_efficacy < ((float) env_vaccination_efficacy[day-1][agent_type])){
            disease_state = RECOVERED;
#ifdef REINFECTION
            unsigned short vaccination_end_of_immunization_days = (unsigned short) max(0.0f, round(cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_VACCINATION_END_OF_IMMUNIZATION_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX)], (int) env_vaccination_end_of_immunization_distr[day-1][agent_type], contacts_id, (float) env_vaccination_end_of_immunization_distr_firstparam[day-1][agent_type], (float) env_vaccination_end_of_immunization_distr_secondparam[day-1][agent_type], false)));
            FLAMEGPU->setVariable<unsigned short>(END_OF_IMMUNIZATION_DAYS, vaccination_end_of_immunization_days);
#endif
            FLAMEGPU->setVariable<unsigned char>(DISEASE_STATE, disease_state);
        }

        // Swabs
        auto env_swab_distr = FLAMEGPU->environment.getMacroProperty<short, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_SWAB_DISTR);
        auto env_swab_distr_firstparam = FLAMEGPU->environment.getMacroProperty<float, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_SWAB_DISTR_FIRSTPARAM);
        auto env_swab_distr_secondparam = FLAMEGPU->environment.getMacroProperty<float, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_SWAB_DISTR_SECONDPARAM);

        int swab_steps = -1;
        if((short) env_swab_distr[day-1][agent_type] != NO_SWAB){
            if(FLAMEGPU->getVariable<int>(SWAB_STEPS) == 0)
                swab_steps = round(cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_SWAB_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX)], (short) env_swab_distr[day-1][agent_type], contacts_id, (float) (STEPS_IN_A_DAY * env_swab_distr_firstparam[day-1][agent_type]), (float) (STEPS_IN_A_DAY * env_swab_distr_secondparam[day-1][agent_type]), true));
            else
                swab_steps = FLAMEGPU->getVariable<int>(SWAB_STEPS);
        }

        FLAMEGPU->setVariable<int>(SWAB_STEPS, swab_steps);

        // Quarantine swabs
        auto env_quarantine_swab_days_distr = FLAMEGPU->environment.getMacroProperty<short, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_QUARANTINE_SWAB_DAYS_DISTR);
        auto env_quarantine_swab_days_distr_firstparam = FLAMEGPU->environment.getMacroProperty<float, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_QUARANTINE_SWAB_DAYS_DISTR_FIRSTPARAM);
        auto env_quarantine_swab_days_distr_secondparam = FLAMEGPU->environment.getMacroProperty<float, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_QUARANTINE_SWAB_DAYS_DISTR_SECONDPARAM);

        const unsigned char quarantine = FLAMEGPU->getVariable<unsigned char>(QUARANTINE);

        if(quarantine > 0){
            int swab_steps = -1;
            if((short) env_quarantine_swab_days_distr[day-1][agent_type] != NO_SWAB){
                if(FLAMEGPU->getVariable<int>(SWAB_STEPS) == 0)
                    swab_steps = round(cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_SWAB_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX)], (short) env_quarantine_swab_days_distr[day-1][agent_type], contacts_id, (float) (STEPS_IN_A_DAY * env_quarantine_swab_days_distr_firstparam[day-1][agent_type]), (float) (STEPS_IN_A_DAY * env_quarantine_swab_days_distr_secondparam[day-1][agent_type]), true));
                else
                    swab_steps = FLAMEGPU->getVariable<int>(SWAB_STEPS);
            }

            FLAMEGPU->setVariable<int>(SWAB_STEPS, swab_steps);
        }
    }

    // Screening
    screening(FLAMEGPU);

#if defined(DEBUG) && !defined(ENSEMBLE)
    printf("5,%d,%d,Ending CUDAContagionScreening for agent with id %d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID));
#endif
    return ALIVE;
}


/** 
    CUDAEvents

    Condition: -

    Handle random events.
*/
FLAMEGPU_AGENT_FUNCTION(CUDAEvents, MessageBucket, MessageBucket) {
#if defined(DEBUG) && !defined(ENSEMBLE)
    printf("5,%d,%d,Ending CUDAEvents for agent with id %d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID));
#endif

    int disease_state = FLAMEGPU->getVariable<unsigned char>(DISEASE_STATE);
    if (disease_state == DIED) {
#if defined(DEBUG) && !defined(ENSEMBLE)
        printf("5,%d,%d,Ending CUDAEvents for agent with id %d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID));
#endif
        return ALIVE;
    }

    auto coord2index = FLAMEGPU->environment.getMacroProperty<short, FLOORS, ENV_DIM_Z, ENV_DIM_X>(COORD2INDEX);
    auto global_resources_counter = FLAMEGPU->environment.getMacroProperty<unsigned int, V>(GLOBAL_RESOURCES_COUNTER);
    auto specific_resources_counter = FLAMEGPU->environment.getMacroProperty<unsigned int, NUMBER_OF_AGENTS_TYPES, V>(SPECIFIC_RESOURCES_COUNTER);
    auto intermediate_target_x = FLAMEGPU->environment.getMacroProperty<float, TOTAL_AGENTS_ESTIMATION, SOLUTION_LENGTH>(INTERMEDIATE_TARGET_X);
    auto intermediate_target_y = FLAMEGPU->environment.getMacroProperty<float, TOTAL_AGENTS_ESTIMATION, SOLUTION_LENGTH>(INTERMEDIATE_TARGET_Y);
    auto intermediate_target_z = FLAMEGPU->environment.getMacroProperty<float, TOTAL_AGENTS_ESTIMATION, SOLUTION_LENGTH>(INTERMEDIATE_TARGET_Z);
    auto env_flow = FLAMEGPU->environment.getMacroProperty<short, NUMBER_OF_AGENTS_TYPES, NUMBER_OF_AGENTS_SUBTYPES, DAYS_IN_A_WEEK, FLOW_LENGTH>(ENV_FLOW);
    auto stay_matrix = FLAMEGPU->environment.getMacroProperty<unsigned int, TOTAL_AGENTS_ESTIMATION, SOLUTION_LENGTH>(STAY);
    auto global_resources = FLAMEGPU->environment.getMacroProperty<unsigned int, V>(GLOBAL_RESOURCES);
    auto specific_resources = FLAMEGPU->environment.getMacroProperty<unsigned int, NUMBER_OF_AGENTS_TYPES, V>(SPECIFIC_RESOURCES);
    auto alternative_resources_area_rand = FLAMEGPU->environment.getMacroProperty<short, NUMBER_OF_AGENTS_TYPES, V>(ALTERNATIVE_RESOURCES_AREA_RAND);
    auto alternative_resources_type_rand = FLAMEGPU->environment.getMacroProperty<short, NUMBER_OF_AGENTS_TYPES, V>(ALTERNATIVE_RESOURCES_TYPE_RAND);

    FLAMEGPU->setVariable<char>(SKIP_FLOW, 0);

    const short contacts_id = FLAMEGPU->getVariable<short>(CONTACTS_ID);
    const short currently_supported = FLAMEGPU->getVariable<short>(CURRENTLY_SUPPORTED);
    const short on_the_way_to_support = FLAMEGPU->getVariable<short>(ON_THE_WAY_TO_SUPPORT);
    const short requested_support = FLAMEGPU->getVariable<short>(REQUESTED_SUPPORT);
    const short agent_type = FLAMEGPU->getVariable<short>(AGENT_TYPE);
    const short agent_subtype = FLAMEGPU->getVariable<short>(AGENT_SUBTYPE);
    const unsigned char quarantine = FLAMEGPU->getVariable<unsigned char>(QUARANTINE);
    const unsigned short flow_index = FLAMEGPU->getVariable<unsigned short>(FLOW_INDEX);
    const float final_target[3] = {FLAMEGPU->getVariable<float, 3>(FINAL_TARGET, 0), FLAMEGPU->getVariable<float, 3>(FINAL_TARGET, 1), FLAMEGPU->getVariable<float, 3>(FINAL_TARGET, 2)};
    
    unsigned short target_index = FLAMEGPU->getVariable<unsigned short>(TARGET_INDEX);
    unsigned short next_index = FLAMEGPU->getVariable<unsigned short>(NEXT_INDEX);
    unsigned char week_day_flow = FLAMEGPU->getVariable<unsigned char>(WEEK_DAY_FLOW);
    unsigned int get_global_resource, get_specific_resource;
    float agent_pos[3] = {FLAMEGPU->getVariable<float>(X), FLAMEGPU->getVariable<float>(Y), FLAMEGPU->getVariable<float>(Z)};
    float intermediate_target[3] = {(float) intermediate_target_x[contacts_id][next_index], (float) intermediate_target_y[contacts_id][next_index], (float) intermediate_target_z[contacts_id][next_index]};

    // By default, the agent does not skip the determined flow
    FLAMEGPU->setVariable<char>(SKIP_FLOW, 0);
 
    // Check if the support agent has reached the agent to support
    if(requested_support == -1 && on_the_way_to_support != -1 && currently_supported == -1){
        // Send a message to it after reaching it
        if((unsigned int) stay_matrix[contacts_id][next_index]){
            FLAMEGPU->setVariable<unsigned short>(CURRENTLY_SUPPORTED, on_the_way_to_support);
            FLAMEGPU->setVariable<short>(ON_THE_WAY_TO_SUPPORT, -1);

            // Send message
            FLAMEGPU->message_out.setVariable<short>(CONTACTS_ID, NUMBER_OF_AGENTS_TYPES + contacts_id);
            FLAMEGPU->message_out.setVariable<int>(REQUEST_ID, -1);
            FLAMEGPU->message_out.setVariable<float>(X, agent_pos[0]);
            FLAMEGPU->message_out.setVariable<float>(Y, agent_pos[1]);
            FLAMEGPU->message_out.setVariable<float>(Z, agent_pos[2]);
            FLAMEGPU->message_out.setVariable<int>(SUPPORT_TIME, -1);

            FLAMEGPU->message_out.setKey(on_the_way_to_support);

            if(next_index != target_index)
                stay_matrix[contacts_id][next_index].exchange(0);

#if defined(DEBUG) && !defined(ENSEMBLE)
            printf("5,%d,%d,Ending CUDAEvents for agent with id %d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID));
#endif
            FLAMEGPU->setVariable<char>(SKIP_FLOW, 1);

            return ALIVE;
        }
    }

    // If an agent is waiting for a support agent, send the initial support message
    if(requested_support != -1 && currently_supported == -1 && on_the_way_to_support == -1 && FLAMEGPU->getVariable<short>(REQUEST_NODE) != -1){
        unsigned int stay_final = stay_matrix[contacts_id][target_index];
        if(stay_final > 1 && FLAMEGPU->getVariable<unsigned char>(IN_AN_EVENT)){
            stay_final = stay_final - 1;
            stay_matrix[contacts_id][target_index].exchange(stay_final);
        }

        FLAMEGPU->message_out.setVariable<short>(CONTACTS_ID, NUMBER_OF_AGENTS_TYPES + contacts_id);
        FLAMEGPU->message_out.setVariable<int>(REQUEST_ID, FLAMEGPU->getVariable<int>(REQUEST_ID));
        FLAMEGPU->message_out.setVariable<float>(X, agent_pos[0]);
        FLAMEGPU->message_out.setVariable<float>(Y, agent_pos[1]);
        FLAMEGPU->message_out.setVariable<float>(Z, agent_pos[2]);
        FLAMEGPU->message_out.setVariable<float>(FINAL_X, FLAMEGPU->environment.getProperty<unsigned short, V>(INDEX2COORDX, FLAMEGPU->getVariable<short>(REQUEST_NODE)));
        FLAMEGPU->message_out.setVariable<float>(FINAL_Y, FLAMEGPU->environment.getProperty<unsigned short, V>(INDEX2COORDY, FLAMEGPU->getVariable<short>(REQUEST_NODE)));
        FLAMEGPU->message_out.setVariable<float>(FINAL_Z, FLAMEGPU->environment.getProperty<unsigned short, V>(INDEX2COORDZ, FLAMEGPU->getVariable<short>(REQUEST_NODE)));
        FLAMEGPU->message_out.setVariable<int>(SUPPORT_TIME, FLAMEGPU->getVariable<int>(REQUEST_TIME));

        FLAMEGPU->message_out.setKey(requested_support);
    }

    // If an agent is waiting for a support agent or an agent is supporting another agent, we skip the rest of the code
    if((requested_support != -1 && currently_supported == -1 && on_the_way_to_support == -1) ||
       (requested_support == -1 && on_the_way_to_support == -1 && currently_supported != -1)){
#if defined(DEBUG) && !defined(ENSEMBLE)
        printf("5,%d,%d,Ending CUDAEvents for agent with id %d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID));
#endif
        FLAMEGPU->setVariable<char>(SKIP_FLOW, 1);

        return ALIVE;
    }

    if(FLAMEGPU->getVariable<unsigned char>(INIT) &&
       !quarantine &&
       !FLAMEGPU->getVariable<unsigned char>(IN_AN_EVENT) &&
       (short) env_flow[agent_type][agent_subtype][week_day_flow][flow_index] != SPAWNROOM &&
       (short) coord2index[(unsigned short)(agent_pos[1]/YOFFSET)][(unsigned short)agent_pos[2]][(unsigned short)agent_pos[0]] != -1 &&
       on_the_way_to_support == -1){
        double random = cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_UNIFORM_0_1_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX)], UNIFORM, contacts_id, 0.0f, 1.0f, false);

        auto env_events = FLAMEGPU->environment.getMacroProperty<short, NUMBER_OF_AGENTS_TYPES, EVENT_LENGTH>(ENV_EVENTS);
        auto env_events_area = FLAMEGPU->environment.getMacroProperty<short, NUMBER_OF_AGENTS_TYPES, EVENT_LENGTH>(ENV_EVENTS_AREA);
        auto env_events_starttime = FLAMEGPU->environment.getMacroProperty<int, NUMBER_OF_AGENTS_TYPES, EVENT_LENGTH>(ENV_EVENTS_STARTTIME);
        auto env_events_endtime = FLAMEGPU->environment.getMacroProperty<int, NUMBER_OF_AGENTS_TYPES, EVENT_LENGTH>(ENV_EVENTS_ENDTIME);
        auto env_events_probability = FLAMEGPU->environment.getMacroProperty<double, NUMBER_OF_AGENTS_TYPES, EVENT_LENGTH>(ENV_EVENTS_PROBABILITY);
        auto env_events_distr = FLAMEGPU->environment.getMacroProperty<short, NUMBER_OF_AGENTS_TYPES, EVENT_LENGTH>(ENV_EVENTS_DISTR);
        auto env_events_distr_firstparam = FLAMEGPU->environment.getMacroProperty<int, NUMBER_OF_AGENTS_TYPES, EVENT_LENGTH>(ENV_EVENTS_DISTR_FIRSTPARAM);
        auto env_events_distr_secondparam = FLAMEGPU->environment.getMacroProperty<int, NUMBER_OF_AGENTS_TYPES, EVENT_LENGTH>(ENV_EVENTS_DISTR_SECONDPARAM);

        double env_events_cdf[EVENT_LENGTH + 1] = {0.0f};
        int env_events_mapping[EVENT_LENGTH + 1] = {-1};

        int step = (FLAMEGPU->getStepCounter() + START_STEP_TIME) % STEPS_IN_A_DAY;

        unsigned int num_events = 1;
        unsigned int i = 0;
        while((short) env_events[agent_type][i] != -1){
            int start_time = (int) env_events_starttime[agent_type][i];
            int end_time = (int) env_events_endtime[agent_type][i];

            if(start_time <= step && step <= end_time){
                env_events_mapping[num_events] = i;
                num_events++;
            }

            i++;
        }

        for(int j = num_events; j > 1; j--){
            if(j == num_events)
                env_events_cdf[j-1] = (double) env_events_probability[agent_type][env_events_mapping[j-1]];
            else
                env_events_cdf[j-1] = env_events_cdf[j] + (double) env_events_probability[agent_type][env_events_mapping[j-1]];
        }
        env_events_cdf[0] = 1.0f;

        int event = env_events_mapping[findLeftmostIndex(random, env_events_cdf, num_events)];

        if(event != -1) {
            short event_node = -1;
            short type_room_event = (short) env_events[agent_type][event];
            short area_room_event = (short) env_events_area[agent_type][event];
            short event_distr = (short) env_events_distr[agent_type][event];
            int event_distr_firstparam = (int) env_events_distr_firstparam[agent_type][event];
            int event_distr_secondparam = (int) env_events_distr_secondparam[agent_type][event];
            float min_separation = numeric_limits<float>::max();
            bool available = false;

            // Searching the nearest room related to the event
            for(const auto& message: FLAMEGPU->message_in(type_room_event)) {

                const unsigned short near_agent_pos[3] = {message.getVariable<unsigned short>(X), message.getVariable<unsigned short>(Y), message.getVariable<unsigned short>(Z)};
                short area_room = message.getVariable<short>(AREA);

                float separation = abs(near_agent_pos[0] - agent_pos[0]) + abs(near_agent_pos[1] - agent_pos[1]) + abs(near_agent_pos[2] - agent_pos[2]);
                if(separation < min_separation && area_room_event == area_room){
                    min_separation = separation;
                    event_node = message.getVariable<short>(GRAPH_NODE);
                }
            }

            short start_node;

            if(next_index != target_index)
                start_node = coord2index[(unsigned short)(intermediate_target[1]/YOFFSET)][(unsigned short)intermediate_target[2]][(unsigned short)intermediate_target[0]];
            else
                start_node = coord2index[(unsigned short)(final_target[1]/YOFFSET)][(unsigned short)final_target[2]][(unsigned short)final_target[0]];

            const short final_node = coord2index[(unsigned short)(final_target[1]/YOFFSET)][(unsigned short)final_target[2]][(unsigned short)final_target[0]];

            short solution_start_event[SOLUTION_LENGTH] = {-1};
            short solution_event_target[SOLUTION_LENGTH] = {-1};

            // Try getting inside the event room (move the resources check when the agent reaches the door of the event room)
            if(event_node != -1){
                get_specific_resource = ++specific_resources_counter[agent_type][event_node];

                if(get_specific_resource <= specific_resources[agent_type][event_node]){
                    get_global_resource = ++global_resources_counter[event_node];

                    if(get_global_resource <= global_resources[event_node]){
                        available = true;
                    }
                    else {
                        --global_resources_counter[event_node];
                        --specific_resources_counter[agent_type][event_node];
                    }
                }
                else {
                    --specific_resources_counter[agent_type][event_node];
                }

                // If the initial room is not avaiable because the resources are over, explore the alternatives:
                if(!available && (short) alternative_resources_type_rand[agent_type][event_node] != -1){
                    // Search another room of the same type and area
                    if((short) alternative_resources_area_rand[agent_type][event_node] == area_room_event && (short) alternative_resources_type_rand[agent_type][event_node] == type_room_event){
                        event_node = findFreeRoomForEventOfTypeAndArea(FLAMEGPU, min_separation, type_room_event, area_room_event, &available);
                    }

                    // Search another room of the alternative
                    else if((short) alternative_resources_type_rand[agent_type][event_node] != type_room_event || (short) alternative_resources_area_rand[agent_type][event_node] != env_events_area){
                        event_node = findFreeRoomForEventOfTypeAndArea(FLAMEGPU, 0, (short) alternative_resources_type_rand[agent_type][event_node], (short) alternative_resources_area_rand[agent_type][event_node], &available);
                    }
                }

                // If the event node is avaiable and the alternative is not skip, then go for the event. Othervise, do nothing
                if(available){
                    a_star(FLAMEGPU, start_node, event_node, solution_start_event);
                    a_star(FLAMEGPU, event_node, final_node, solution_event_target);

                    unsigned int event_time_random = (unsigned int) cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_EVENT_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX)], event_distr, contacts_id, (float) event_distr_firstparam, (float) event_distr_secondparam, true);
                    unsigned int final_stay;

                    if(event_time_random < (unsigned int) stay_matrix[contacts_id][target_index]){
                        final_stay = (unsigned int) stay_matrix[contacts_id][target_index] - event_time_random;
                    }
                    else{
                        final_stay = 1;
                    }


                    update_targets(FLAMEGPU, solution_start_event, &target_index, true, event_time_random);
                    update_targets(FLAMEGPU, solution_event_target, &target_index, false, final_stay);

                    FLAMEGPU->setVariable<unsigned char>(IN_AN_EVENT, 1);
                    FLAMEGPU->setVariable<short>(ACTUAL_EVENT_NODE, event_node);

                    auto env_events_agentlinked = FLAMEGPU->environment.getMacroProperty<short, NUMBER_OF_AGENTS_TYPES, EVENT_LENGTH>(ENV_EVENTS_AGENTLINKED);
                    auto env_events_agentlinked_type = FLAMEGPU->environment.getMacroProperty<short, NUMBER_OF_AGENTS_TYPES, EVENT_LENGTH>(ENV_EVENTS_AGENTLINKED_TYPE);
                    auto env_events_agentlinked_timeout = FLAMEGPU->environment.getMacroProperty<short, NUMBER_OF_AGENTS_TYPES, EVENT_LENGTH>(ENV_EVENTS_AGENTLINKED_TIMEOUT);
                    auto env_events_agentlinked_timeout_behave = FLAMEGPU->environment.getMacroProperty<unsigned char, NUMBER_OF_AGENTS_TYPES, EVENT_LENGTH>(ENV_EVENTS_AGENTLINKED_TIMEOUT_BEHAVE);

                    short agentlinked = (short) env_events_agentlinked[agent_type][event];
                    short agentlinked_type = (short) env_events_agentlinked_type[agent_type][event];

                    // Handle new support, if necessary
                    if(agentlinked != -1){
                        if(requested_support == -1){
                            FLAMEGPU->setVariable<short>(REQUESTED_SUPPORT, agentlinked);
                            FLAMEGPU->setVariable<char>(REQUESTED_TYPE, (char) agentlinked_type);

                            auto support_requests = FLAMEGPU->environment.getMacroProperty<unsigned int, NUMBER_OF_AGENTS_TYPES, 2>(SUPPORT_REQUESTS);

                            unsigned int request_id = ++support_requests[agentlinked][0];

                            FLAMEGPU->setVariable<int>(REQUEST_ID, (int) request_id);
                            FLAMEGPU->setVariable<short>(REQUEST_NODE, event_node);
                            FLAMEGPU->setVariable<int>(REQUEST_TIME, (unsigned char) agentlinked_type == ACCOMPANIMENT_ONLY ? 0: event_time_random);
                            FLAMEGPU->setVariable<short>(REQUEST_WAITING_TIME, (short) env_events_agentlinked_timeout[agent_type][event]);
                            FLAMEGPU->setVariable<unsigned char>(REQUEST_WAITING_TIME_BEHAVE, (unsigned char) env_events_agentlinked_timeout_behave[agent_type][event]);

                            FLAMEGPU->message_out.setVariable<short>(CONTACTS_ID, NUMBER_OF_AGENTS_TYPES + contacts_id);
                            FLAMEGPU->message_out.setVariable<int>(REQUEST_ID, (int) request_id);
                            FLAMEGPU->message_out.setVariable<float>(X, agent_pos[0]);
                            FLAMEGPU->message_out.setVariable<float>(Y, agent_pos[1]);
                            FLAMEGPU->message_out.setVariable<float>(Z, agent_pos[2]);
                            FLAMEGPU->message_out.setVariable<float>(FINAL_X, FLAMEGPU->environment.getProperty<unsigned short, V>(INDEX2COORDX, event_node));
                            FLAMEGPU->message_out.setVariable<float>(FINAL_Y, FLAMEGPU->environment.getProperty<unsigned short, V>(INDEX2COORDY, event_node));
                            FLAMEGPU->message_out.setVariable<float>(FINAL_Z, FLAMEGPU->environment.getProperty<unsigned short, V>(INDEX2COORDZ, event_node));
                            FLAMEGPU->message_out.setVariable<int>(SUPPORT_TIME, agentlinked_type == ACCOMPANIMENT_ONLY ? 0: event_time_random);

                            FLAMEGPU->message_out.setKey(agentlinked);
                        }
                        else{
                            // The agent is already supported in the determined flow; we extend the support to the event
                            FLAMEGPU->setVariable<short>(REQUESTED_SUPPORT_EVENT_WITH_FLOW, 1);
                            FLAMEGPU->setVariable<short>(SUPPORT_TIME_EVENT, event_time_random);
                        }
                    }

                    if(agent_pos[1] == INVISIBLE_AGENT_Y)
                        printf("0,%d,%d,%d,%d,%f,%f,%f,%d,-1\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID), FLAMEGPU->getVariable<short>(AGENT_TYPE), agent_pos[0], INVISIBLE_AGENT_Y, agent_pos[2], FLAMEGPU->getVariable<unsigned char>(DISEASE_STATE));
                    else
                        printf("0,%d,%d,%d,%d,%f,%f,%f,%d,%d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID), FLAMEGPU->getVariable<short>(AGENT_TYPE), agent_pos[0], agent_pos[1], agent_pos[2], FLAMEGPU->getVariable<unsigned char>(DISEASE_STATE), (short) coord2index[(unsigned short)(agent_pos[1]/YOFFSET)][(unsigned short)agent_pos[2]][(unsigned short)agent_pos[0]]);

                    FLAMEGPU->setVariable<char>(CAN_MOVE, 1);
                    FLAMEGPU->setVariable<char>(SKIP_FLOW, 1);

#if defined(DEBUG) && !defined(ENSEMBLE)
                    printf("5,%d,%d,Ending CUDAEvents for agent with id %d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID));
#endif
                    return ALIVE;
                }

            }

            FLAMEGPU->setVariable<char>(CAN_MOVE, 1);
#if defined(DEBUG) && !defined(ENSEMBLE)
            printf("5,%d,%d,Ending CUDAEvents for agent with id %d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID));
#endif
            return ALIVE;
        }
    }

#if defined(DEBUG) && !defined(ENSEMBLE)
    printf("5,%d,%d,Ending CUDAEvents for agent with id %d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID));
#endif
    return ALIVE;

}


/** 
    CUDAMovePedestrian

    Condition: -

    Handle determined flow.
*/
FLAMEGPU_AGENT_FUNCTION(CUDAMovePedestrian, MessageBucket, MessageBucket) {
#if defined(DEBUG) && !defined(ENSEMBLE)
    printf("5,%d,%d,Beginning CUDAMovePedestrian for agent with id %d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID));
#endif
    // If the agent is in an event
    if (FLAMEGPU->getVariable<char>(SKIP_FLOW)) {
#if defined(DEBUG) && !defined(ENSEMBLE)
        printf("5,%d,%d,Ending CUDAMovePedestrian for agent with id %d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID));
#endif
        return ALIVE;
    }

    // Move pedestrian
    auto env_flow = FLAMEGPU->environment.getMacroProperty<short, NUMBER_OF_AGENTS_TYPES, NUMBER_OF_AGENTS_SUBTYPES, DAYS_IN_A_WEEK, FLOW_LENGTH>(ENV_FLOW);
    auto env_flow_area = FLAMEGPU->environment.getMacroProperty<short, NUMBER_OF_AGENTS_TYPES, NUMBER_OF_AGENTS_SUBTYPES, DAYS_IN_A_WEEK, FLOW_LENGTH>(ENV_FLOW_AREA);
    auto counters = FLAMEGPU->environment.getMacroProperty<unsigned int, NUM_COUNTERS>(COUNTERS);
    auto stay_matrix = FLAMEGPU->environment.getMacroProperty<unsigned int, TOTAL_AGENTS_ESTIMATION, SOLUTION_LENGTH>(STAY);
    auto global_resources_counter = FLAMEGPU->environment.getMacroProperty<unsigned int, V>(GLOBAL_RESOURCES_COUNTER);
    auto specific_resources_counter = FLAMEGPU->environment.getMacroProperty<unsigned int, NUMBER_OF_AGENTS_TYPES, V>(SPECIFIC_RESOURCES_COUNTER);
    auto spawnrooms_areas_ids = FLAMEGPU->environment.getMacroProperty<unsigned short, NUM_AREAS, NUM_SPAWNROOM + 1>(SPAWNROOMS_AREAS_IDS);
    auto coord2index = FLAMEGPU->environment.getMacroProperty<short, FLOORS, ENV_DIM_Z, ENV_DIM_X>(COORD2INDEX);

    const unsigned char agent_with_a_rate = FLAMEGPU->getVariable<unsigned char>(AGENT_WITH_A_RATE);
    const unsigned short flow_index = FLAMEGPU->getVariable<unsigned short>(FLOW_INDEX);
    const unsigned char quarantine = FLAMEGPU->getVariable<unsigned char>(QUARANTINE);
    const int room_for_quarantine_index = FLAMEGPU->getVariable<int>(ROOM_FOR_QUARANTINE_INDEX);
    const short agent_type = FLAMEGPU->getVariable<short>(AGENT_TYPE);
    const short agent_subtype = FLAMEGPU->getVariable<short>(AGENT_SUBTYPE);
    const short contacts_id = FLAMEGPU->getVariable<short>(CONTACTS_ID);
    const float final_target[3] = {FLAMEGPU->getVariable<float, 3>(FINAL_TARGET, 0), FLAMEGPU->getVariable<float, 3>(FINAL_TARGET, 1), FLAMEGPU->getVariable<float, 3>(FINAL_TARGET, 2)};
    const short arrival_node = coord2index[(unsigned short)(final_target[1]/YOFFSET)][(unsigned short)final_target[2]][(unsigned short)final_target[0]];

    unsigned short next_index = FLAMEGPU->getVariable<unsigned short>(NEXT_INDEX);
    unsigned short target_index = FLAMEGPU->getVariable<unsigned short>(TARGET_INDEX);
    unsigned char week_day_flow = FLAMEGPU->getVariable<unsigned char>(WEEK_DAY_FLOW);
    unsigned int stay = (unsigned int) stay_matrix[contacts_id][next_index];
    float agent_pos[3] = {FLAMEGPU->getVariable<float>(X), FLAMEGPU->getVariable<float>(Y), FLAMEGPU->getVariable<float>(Z)};
    int disease_state = FLAMEGPU->getVariable<unsigned char>(DISEASE_STATE);
    // short solution[SOLUTION_LENGTH] = {-1};

    // 2. Check Arrival at Spawnroom
    if(FLAMEGPU->getVariable<unsigned char>(INIT) && CHECK_IS_SPAWNROOM(arrival_node) && next_index == target_index && (flow_index > 1 || FLAMEGPU->getVariable<unsigned char>(JUST_EXITED_FROM_QUARANTINE) || CHECK_IS_SPAWNROOM(room_for_quarantine_index))) {
        bool is_terminal_exit = ((short) env_flow[agent_type][agent_subtype][week_day_flow][flow_index + 1] == -1 &&
                                 (short) env_flow[agent_type][agent_subtype][week_day_flow][flow_index] == SPAWNROOM) ||
                                CHECK_IS_SPAWNROOM(room_for_quarantine_index) ||
                                FLAMEGPU->getVariable<unsigned char>(JUST_EXITED_FROM_QUARANTINE);

        printf("0,%d,%d,%d,%d,%f,%f,%f,%d,-1\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), contacts_id, agent_type, agent_pos[0], INVISIBLE_AGENT_Y, agent_pos[2], disease_state);

        if (is_terminal_exit) {
            if(agent_with_a_rate && (short) env_flow[agent_type][agent_subtype][week_day_flow][flow_index + 1] == -1){
                counters[COUNTERS_KILLED_AGENTS_WITH_RATE]++;


#if defined(DEBUG) && !defined(ENSEMBLE)
                printf("5,%d,%d,Ending CUDAMovePedestrian for agent with id %d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID));
#endif
                return DEAD;
            }

            FLAMEGPU->setVariable<unsigned char>(INIT, 0);
            FLAMEGPU->setVariable<float>(Y, INVISIBLE_AGENT_Y);
            FLAMEGPU->setVariable<char>(CAN_MOVE, 0);

            if(!agent_with_a_rate && (short) env_flow[agent_type][agent_subtype][week_day_flow][flow_index + 1] == -1 && !FLAMEGPU->getVariable<unsigned char>(JUST_EXITED_FROM_QUARANTINE))
                update_flow(FLAMEGPU, false);

            FLAMEGPU->setVariable<unsigned char>(JUST_EXITED_FROM_QUARANTINE, 0);
#if defined(DEBUG) && !defined(ENSEMBLE)
            printf("5,%d,%d,Ending CUDAMovePedestrian for agent with id %d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID));
#endif

            return ALIVE;
        } else if((short) env_flow[agent_type][agent_subtype][week_day_flow][flow_index] == SPAWNROOM){

            FLAMEGPU->setVariable<unsigned char>(INIT, 0);
            FLAMEGPU->setVariable<float>(Y, INVISIBLE_AGENT_Y);
            FLAMEGPU->setVariable<char>(CAN_MOVE, 0);

            return ALIVE;
        }
    }

    bool just_finished_event = false;

    // Decrement stay and eventually take the next (or first) destination in the agent's flow
    if(stay){
        stay = stay - 1;
        stay_matrix[contacts_id][next_index].exchange(stay);

        if(!stay && next_index == target_index && quarantine){
            exit_from_quarantine(FLAMEGPU);

#if defined(DEBUG) && !defined(ENSEMBLE)
            printf("5,%d,%d,Ending CUDAMovePedestrian for agent with id %d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID));
#endif
            return ALIVE;
        }

        if(next_index == target_index && !stay && FLAMEGPU->getVariable<unsigned char>(IN_AN_EVENT) == 1 && FLAMEGPU->getVariable<short>(ACTUAL_EVENT_NODE) != -1){
            FLAMEGPU->setVariable<unsigned char>(IN_AN_EVENT, 2);
            just_finished_event = true;
            short event_node = FLAMEGPU->getVariable<short>(ACTUAL_EVENT_NODE);

            FLAMEGPU->setVariable<unsigned char>(IN_AN_EVENT, 2);
            FLAMEGPU->setVariable<short>(ACTUAL_EVENT_NODE, -1);
            FLAMEGPU->setVariable<short>(REQUESTED_SUPPORT_EVENT_WITH_FLOW, -1);
            
            just_finished_event = true;
            --global_resources_counter[event_node];
            --specific_resources_counter[agent_type][event_node];
        }

        if(!stay)
            if(agent_pos[1] == INVISIBLE_AGENT_Y)
                printf("0,%d,%d,%d,%d,%f,%f,%f,%d,-1\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID), FLAMEGPU->getVariable<short>(AGENT_TYPE), agent_pos[0], INVISIBLE_AGENT_Y, agent_pos[2], FLAMEGPU->getVariable<unsigned char>(DISEASE_STATE));
            else
                printf("0,%d,%d,%d,%d,%f,%f,%f,%d,%d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID), FLAMEGPU->getVariable<short>(AGENT_TYPE), agent_pos[0], agent_pos[1], agent_pos[2], FLAMEGPU->getVariable<unsigned char>(DISEASE_STATE), (short) coord2index[(unsigned short)(agent_pos[1]/YOFFSET)][(unsigned short)agent_pos[2]][(unsigned short)agent_pos[0]]);

        if(!stay && next_index == target_index && (short) env_flow[agent_type][agent_subtype][week_day_flow][flow_index] != -1){
            if(!FLAMEGPU->getVariable<unsigned char>(INIT)){
                FLAMEGPU->setVariable<unsigned char>(INIT, 1);

                unsigned short spawnroom_id = GET_SPAWNROOM_ID_FOR_VECTORS((unsigned short) spawnrooms_areas_ids[(short) env_flow_area[agent_type][agent_subtype][week_day_flow][flow_index]][(unsigned short) (cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_UNIFORM_0_1_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX)], UNIFORM, contacts_id, 1.0f, (unsigned short) spawnrooms_areas_ids[(short) env_flow_area[agent_type][agent_subtype][week_day_flow][flow_index]][0], false))]);

                float x = cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_OFFSET_X_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX)], UNIFORM, contacts_id, FLAMEGPU->environment.getProperty<float, NUM_SPAWNROOM * 4>(EXTERN_RANGES, spawnroom_id * 2), FLAMEGPU->environment.getProperty<float, NUM_SPAWNROOM * 4>(EXTERN_RANGES, (spawnroom_id * 2) + 1), false);
                float y = FLAMEGPU->environment.getProperty<unsigned short, NUM_SPAWNROOM + 1>(ENTRANCE_Y_COORDS, spawnroom_id);
                float z = cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_OFFSET_Z_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX)], UNIFORM, contacts_id, FLAMEGPU->environment.getProperty<float, NUM_SPAWNROOM * 4>(EXTERN_RANGES, (spawnroom_id * 2) + 2 * NUM_SPAWNROOM), FLAMEGPU->environment.getProperty<float, NUM_SPAWNROOM * 4>(EXTERN_RANGES, (spawnroom_id * 2) + 2 * NUM_SPAWNROOM + 1), false);

                FLAMEGPU->setVariable<float>(X, x);
                FLAMEGPU->setVariable<float>(Y, y);
                FLAMEGPU->setVariable<float>(Z, z);

                agent_pos[0] = x;
                agent_pos[1] = y;
                agent_pos[2] = z;

                FLAMEGPU->setVariable<float, 3>(FINAL_TARGET, 0, x);
                FLAMEGPU->setVariable<float, 3>(FINAL_TARGET, 1, y);
                FLAMEGPU->setVariable<float, 3>(FINAL_TARGET, 2, z);
            }

            int flow_stay = 1;

            const short start_node = coord2index[(unsigned short)(final_target[1]/YOFFSET)][(unsigned short)final_target[2]][(unsigned short)final_target[0]];
            const short start_node_type = FLAMEGPU->environment.getProperty<short, V>(NODE_TYPE, start_node);

            if(!CHECK_IS_SPAWNROOM(start_node) && start_node_type != WAITINGROOM) {
                --global_resources_counter[start_node];
                --specific_resources_counter[agent_type][start_node];
            }

            short actual_node = FLAMEGPU->getVariable<short>(ACTUAL_NODE);
            short actual_node_object = FLAMEGPU->getVariable<short>(ACTUAL_NODE_OBJECT);

            if(actual_node_object != -1){
                --rooms_resources_global_objects_counter[agent_type][actual_node][actual_node_object];
                --rooms_resources_specific_objects_counter[agent_type][actual_node][actual_node_object];

                FLAMEGPU->setVariable<short>(ACTUAL_NODE, -1);
                FLAMEGPU->setVariable<short>(ACTUAL_NODE_STAY, -1);
                FLAMEGPU->setVariable<short>(ACTUAL_NODE_OBJECT, -1);
            }

            if (FLAMEGPU->getVariable<unsigned char>(IN_AN_EVENT) == 2 && !just_finished_event) {
                FLAMEGPU->setVariable<unsigned char>(IN_AN_EVENT, 0);
            }

            if(disease_state == DIED){
                if(agent_with_a_rate)
                    counters[COUNTERS_KILLED_AGENTS_WITH_RATE]++;

#if defined(DEBUG) && !defined(ENSEMBLE)
                printf("5,%d,%d,Ending CUDAMovePedestrian for agent with id %d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID));
#endif
                return DEAD;
            }

            bool available = false;

            const short final_node = take_new_destination_flow(FLAMEGPU, &flow_stay, start_node, &available);

            // Handle agent linked with an other agent
            auto env_flow_agentlinked = FLAMEGPU->environment.getMacroProperty<short, NUMBER_OF_AGENTS_TYPES, NUMBER_OF_AGENTS_SUBTYPES, DAYS_IN_A_WEEK, FLOW_LENGTH>(ENV_FLOW_AGENTLINKED);
            auto env_flow_agentlinked_type = FLAMEGPU->environment.getMacroProperty<short, NUMBER_OF_AGENTS_TYPES, NUMBER_OF_AGENTS_SUBTYPES, DAYS_IN_A_WEEK, FLOW_LENGTH>(ENV_FLOW_AGENTLINKED_TYPE);
            auto env_flow_agentlinked_timeout = FLAMEGPU->environment.getMacroProperty<short, NUMBER_OF_AGENTS_TYPES, NUMBER_OF_AGENTS_SUBTYPES, DAYS_IN_A_WEEK, FLOW_LENGTH>(ENV_FLOW_AGENTLINKED_TIMEOUT);
            auto env_flow_agentlinked_timeout_behave = FLAMEGPU->environment.getMacroProperty<unsigned char, NUMBER_OF_AGENTS_TYPES, NUMBER_OF_AGENTS_SUBTYPES, DAYS_IN_A_WEEK, FLOW_LENGTH>(ENV_FLOW_AGENTLINKED_TIMEOUT_BEHAVE);

            // Handle new support, if necessary
            short agentlinked = (short) env_flow_agentlinked[agent_type][agent_subtype][week_day_flow][flow_index + 1];
            short agentlinked_type = (short) env_flow_agentlinked_type[agent_type][agent_subtype][week_day_flow][flow_index + 1];
            if(agentlinked != -1 && (available && FLAMEGPU->getVariable<unsigned char>(WAITING_ROOM_FLAG) == OUTSIDE_WAITING_ROOM)){
                FLAMEGPU->setVariable<short>(REQUESTED_SUPPORT, agentlinked);
                FLAMEGPU->setVariable<char>(REQUESTED_TYPE, (char) agentlinked_type);

                auto support_requests = FLAMEGPU->environment.getMacroProperty<unsigned int, NUMBER_OF_AGENTS_TYPES, 2>(SUPPORT_REQUESTS);

                unsigned int request_id = ++support_requests[agentlinked][0];

                FLAMEGPU->setVariable<int>(REQUEST_ID, (int) request_id);
                FLAMEGPU->setVariable<short>(REQUEST_NODE, final_node);
                FLAMEGPU->setVariable<int>(REQUEST_TIME, agentlinked_type == ACCOMPANIMENT_ONLY ? 0: flow_stay);
                FLAMEGPU->setVariable<short>(REQUEST_WAITING_TIME, (short) env_flow_agentlinked_timeout[agent_type][agent_subtype][week_day_flow][flow_index + 1]);
                FLAMEGPU->setVariable<unsigned char>(REQUEST_WAITING_TIME_BEHAVE, (unsigned char) env_flow_agentlinked_timeout_behave[agent_type][agent_subtype][week_day_flow][flow_index + 1]);

                FLAMEGPU->message_out.setVariable<short>(CONTACTS_ID, NUMBER_OF_AGENTS_TYPES + contacts_id);
                FLAMEGPU->message_out.setVariable<int>(REQUEST_ID, (int) request_id);
                FLAMEGPU->message_out.setVariable<float>(X, agent_pos[0]);
                FLAMEGPU->message_out.setVariable<float>(Y, agent_pos[1]);
                FLAMEGPU->message_out.setVariable<float>(Z, agent_pos[2]);
                FLAMEGPU->message_out.setVariable<float>(FINAL_X, FLAMEGPU->environment.getProperty<unsigned short, V>(INDEX2COORDX, final_node));
                FLAMEGPU->message_out.setVariable<float>(FINAL_Y, FLAMEGPU->environment.getProperty<unsigned short, V>(INDEX2COORDY, final_node));
                FLAMEGPU->message_out.setVariable<float>(FINAL_Z, FLAMEGPU->environment.getProperty<unsigned short, V>(INDEX2COORDZ, final_node));
                FLAMEGPU->message_out.setVariable<int>(SUPPORT_TIME, agentlinked_type == ACCOMPANIMENT_ONLY ? 0: flow_stay);

                FLAMEGPU->message_out.setKey(agentlinked);
            }

            FLAMEGPU->setVariable<short>(ACTUAL_NODE, final_node);
            FLAMEGPU->setVariable<short>(ACTUAL_NODE_STAY, flow_stay);

            room2door_logic(FLAMEGPU, start_node);

            FLAMEGPU->setVariable<char>(CAN_MOVE, 1);
            FLAMEGPU->setVariable<char>(SKIP_FLOW, 1);
        }

#if defined(DEBUG) && !defined(ENSEMBLE)
        printf("5,%d,%d,Ending CUDAMovePedestrian for agent with id %d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID));
#endif
        return ALIVE;
    }

    if (FLAMEGPU->getVariable<float>(Y) != INVISIBLE_AGENT_Y) {
        FLAMEGPU->setVariable<char>(CAN_MOVE, 1);
    } else {
        FLAMEGPU->setVariable<char>(CAN_MOVE, 0);
    }

#if defined(DEBUG) && !defined(ENSEMBLE)
    printf("5,%d,%d,Ending CUDAMovePedestrian for agent with id %d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID));
#endif
    return ALIVE;
}


/**
    handleSupportRequest

    Condition: initCondition

    Start to support an agent, if necessary.
*/
FLAMEGPU_AGENT_FUNCTION(handleSupportRequest, MessageBucket, MessageNone) {
#if defined(DEBUG) && !defined(ENSEMBLE)
    printf("5,%d,%d,Starting handleSupportRequest for agent with id %d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID));
#endif
    auto stay_matrix = FLAMEGPU->environment.getMacroProperty<unsigned int, TOTAL_AGENTS_ESTIMATION, SOLUTION_LENGTH>(STAY);
    auto coord2index = FLAMEGPU->environment.getMacroProperty<short, FLOORS, ENV_DIM_Z, ENV_DIM_X>(COORD2INDEX);

    const short currently_supported = FLAMEGPU->getVariable<short>(CURRENTLY_SUPPORTED);
    const short on_the_way_to_support = FLAMEGPU->getVariable<short>(ON_THE_WAY_TO_SUPPORT);
    const short requested_support = FLAMEGPU->getVariable<short>(REQUESTED_SUPPORT);
    const short agent_type = FLAMEGPU->getVariable<short>(AGENT_TYPE);
    const short contacts_id = FLAMEGPU->getVariable<short>(CONTACTS_ID);
    const unsigned short next_index = FLAMEGPU->getVariable<unsigned short>(NEXT_INDEX);
    const unsigned char in_an_event = FLAMEGPU->getVariable<unsigned char>(IN_AN_EVENT);

    unsigned short target_index = FLAMEGPU->getVariable<unsigned short>(TARGET_INDEX);

    // Check if there is at least one support request
    if(currently_supported == -1 && on_the_way_to_support == -1 && requested_support == -1 && !in_an_event){
        auto support_requests = FLAMEGPU->environment.getMacroProperty<unsigned int, NUMBER_OF_AGENTS_TYPES, 2>(SUPPORT_REQUESTS);

        unsigned int total_requests = support_requests[agent_type][0];
        unsigned int request_id = ++support_requests[agent_type][1];
        bool found = false;

        if(total_requests >= request_id){
            auto messages = FLAMEGPU->message_in(agent_type);
            auto interested_message = messages.begin();

            while(interested_message != messages.end()){
                const int message_request_id = (*interested_message).getVariable<int>(REQUEST_ID);

                if(message_request_id != (int) request_id)
                    interested_message++;
                else{
                    found = true;
                    break;
                }
            }

            if(!found){
#if defined(DEBUG) && !defined(ENSEMBLE)
                printf("5,%d,%d,Ending handleSupportRequest for agent with id %d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID));
#endif
                return ALIVE;
            }

            auto coord2index = FLAMEGPU->environment.getMacroProperty<short, FLOORS, ENV_DIM_Z, ENV_DIM_X>(COORD2INDEX);

            const float final_target[3] = {FLAMEGPU->getVariable<float, 3>(FINAL_TARGET, 0), FLAMEGPU->getVariable<float, 3>(FINAL_TARGET, 1), FLAMEGPU->getVariable<float, 3>(FINAL_TARGET, 2)};
            const short start_node = coord2index[(unsigned short)(final_target[1]/YOFFSET)][(unsigned short)final_target[2]][(unsigned short)final_target[0]];
            const short target_node = coord2index[(unsigned short)((*interested_message).getVariable<float>(Y)/YOFFSET)][(unsigned short)(*interested_message).getVariable<float>(Z)][(unsigned short)(*interested_message).getVariable<float>(X)];
            const short support_node = coord2index[(unsigned short)((*interested_message).getVariable<float>(FINAL_Y)/YOFFSET)][(unsigned short)(*interested_message).getVariable<float>(FINAL_Z)][(unsigned short)(*interested_message).getVariable<float>(FINAL_X)];
            const short final_node = start_node;

            short solution_start_support[SOLUTION_LENGTH] = {-1};
            short solution_support_final[SOLUTION_LENGTH] = {-1};

            int support_stay = 1;
            int final_stay = (unsigned int) stay_matrix[contacts_id][target_index] - (*interested_message).getVariable<int>(SUPPORT_TIME);

            final_stay = final_stay > 0 ? final_stay: 1;

            a_star(FLAMEGPU, start_node, target_node, solution_start_support);
            a_star(FLAMEGPU, support_node, final_node, solution_support_final);

            update_targets(FLAMEGPU, solution_start_support, &target_index, true, support_stay);
            update_targets(FLAMEGPU, solution_support_final, &target_index, false, final_stay);

            FLAMEGPU->setVariable<unsigned short>(ON_THE_WAY_TO_SUPPORT, (unsigned short) (*interested_message).getVariable<short>(CONTACTS_ID));
        }
        else{
            --support_requests[agent_type][1];
        }
    }

#if defined(DEBUG) && !defined(ENSEMBLE)
    printf("5,%d,%d,Ending handleSupportRequest for agent with id %d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID));
#endif
    return ALIVE;
}


/**
    waitingForSupport

    Condition: initCondition

    Waiting for support from an agent.
*/
FLAMEGPU_AGENT_FUNCTION(waitingForSupport, MessageBucket, MessageNone) {
#if defined(DEBUG) && !defined(ENSEMBLE)
    printf("5,%d,%d,Starting waitingForSupport for agent with id %d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID));
#endif
    auto stay_matrix = FLAMEGPU->environment.getMacroProperty<unsigned int, TOTAL_AGENTS_ESTIMATION, SOLUTION_LENGTH>(STAY);
    auto coord2index = FLAMEGPU->environment.getMacroProperty<short, FLOORS, ENV_DIM_Z, ENV_DIM_X>(COORD2INDEX);

    const short currently_supported = FLAMEGPU->getVariable<short>(CURRENTLY_SUPPORTED);
    const short on_the_way_to_support = FLAMEGPU->getVariable<short>(ON_THE_WAY_TO_SUPPORT);
    const short requested_support = FLAMEGPU->getVariable<short>(REQUESTED_SUPPORT);
    const short contacts_id = FLAMEGPU->getVariable<short>(CONTACTS_ID);
    const short agent_type = FLAMEGPU->getVariable<short>(AGENT_TYPE);

    unsigned short next_index = FLAMEGPU->getVariable<unsigned short>(NEXT_INDEX);
    unsigned short target_index = FLAMEGPU->getVariable<unsigned short>(TARGET_INDEX);
    unsigned int stay = (unsigned int) stay_matrix[contacts_id][next_index];

    // The agent which requested support is waiting for the support agent
    if(currently_supported == -1 && requested_support != -1 && on_the_way_to_support == -1){
        auto support_requests = FLAMEGPU->environment.getMacroProperty<unsigned int, NUMBER_OF_AGENTS_TYPES, 2>(SUPPORT_REQUESTS);

        int request_id = FLAMEGPU->getVariable<int>(REQUEST_ID);

        if(support_requests[requested_support][1] >= request_id){
            auto messages = FLAMEGPU->message_in(NUMBER_OF_AGENTS_TYPES + contacts_id);
            auto interested_message = messages.begin();

            // The agent received the start message from the support agent
            if(interested_message != messages.end()){
                FLAMEGPU->setVariable<unsigned short>(CURRENTLY_SUPPORTED, (unsigned short) (*interested_message).getVariable<short>(CONTACTS_ID));

                FLAMEGPU->setVariable<int>(REQUEST_ID, -1);
                FLAMEGPU->setVariable<int>(REQUEST_TIME, -1);
                FLAMEGPU->setVariable<short>(REQUEST_NODE, -1);
            }
        }
        else{
            int request_waiting_time = FLAMEGPU->getVariable<int>(REQUEST_WAITING_TIME);
            int request_waiting_time_behave = FLAMEGPU->getVariable<int>(REQUEST_WAITING_TIME_BEHAVE);

            if(request_waiting_time <= 0){
                FLAMEGPU->setVariable<int>(REQUEST_ID, -1);
                FLAMEGPU->setVariable<int>(REQUEST_TIME, -1);
                FLAMEGPU->setVariable<short>(REQUEST_NODE, -1);
                FLAMEGPU->setVariable<short>(REQUESTED_SUPPORT, -1);
                FLAMEGPU->setVariable<char>(REQUESTED_TYPE, -1);

                FLAMEGPU->setVariable<int>(REQUEST_WAITING_TIME, -1);

                if(request_waiting_time_behave == SKIP_PIECE_OF_FLOW){
                    const float final_target[3] = {FLAMEGPU->getVariable<float, 3>(FINAL_TARGET, 0), FLAMEGPU->getVariable<float, 3>(FINAL_TARGET, 1), FLAMEGPU->getVariable<float, 3>(FINAL_TARGET, 2)};
                    const short target_node = coord2index[(unsigned short)(final_target[1]/YOFFSET)][(unsigned short)final_target[2]][(unsigned short)final_target[0]];

                    short solution[SOLUTION_LENGTH] = {-1};

                    a_star(FLAMEGPU, target_node, target_node, solution);
                    update_targets(FLAMEGPU, solution, &target_index, true, 1);
                }
                else{
                    if(request_waiting_time_behave == SKIP_EVENT){
                        auto intermediate_target_x = FLAMEGPU->environment.getMacroProperty<float, TOTAL_AGENTS_ESTIMATION, SOLUTION_LENGTH>(INTERMEDIATE_TARGET_X);
                        auto intermediate_target_y = FLAMEGPU->environment.getMacroProperty<float, TOTAL_AGENTS_ESTIMATION, SOLUTION_LENGTH>(INTERMEDIATE_TARGET_Y);
                        auto intermediate_target_z = FLAMEGPU->environment.getMacroProperty<float, TOTAL_AGENTS_ESTIMATION, SOLUTION_LENGTH>(INTERMEDIATE_TARGET_Z);

                        const float intermediate_target[3] = {(float) intermediate_target_x[contacts_id][next_index], (float) intermediate_target_y[contacts_id][next_index], (float) intermediate_target_z[contacts_id][next_index]};
                        const float final_target[3] = {FLAMEGPU->getVariable<float, 3>(FINAL_TARGET, 0), FLAMEGPU->getVariable<float, 3>(FINAL_TARGET, 1), FLAMEGPU->getVariable<float, 3>(FINAL_TARGET, 2)};
                        const short start_node = coord2index[(unsigned short)(intermediate_target[1]/YOFFSET)][(unsigned short)(intermediate_target[2])][(unsigned short)(intermediate_target[0])];
                        const short target_node = coord2index[(unsigned short)(final_target[1]/YOFFSET)][(unsigned short)final_target[2]][(unsigned short)final_target[0]];

                        short solution[SOLUTION_LENGTH] = {-1};

                        a_star(FLAMEGPU, start_node, target_node, solution);
                        update_targets(FLAMEGPU, solution, &target_index, true, stay_matrix[contacts_id][target_index]);

                        FLAMEGPU->setVariable<unsigned char>(IN_AN_EVENT, 0);
                        FLAMEGPU->setVariable<short>(ACTUAL_EVENT_NODE, -1);
                    }
                }

                FLAMEGPU->setVariable<int>(REQUEST_WAITING_TIME_BEHAVE, -1);
            }
            else{
                FLAMEGPU->setVariable<int>(REQUEST_WAITING_TIME, request_waiting_time - 1);
            }
        }
    }

#if defined(DEBUG) && !defined(ENSEMBLE)
    printf("5,%d,%d,Ending waitingForSupport for agent with id %d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID));
#endif
    return ALIVE;
}


/**
    beingSupported

    Condition: initCondition

    Being supported by an agent.
*/
FLAMEGPU_AGENT_FUNCTION(beingSupported, MessageNone, MessageBucket) {
#if defined(DEBUG) && !defined(ENSEMBLE)
    printf("5,%d,%d,Starting beingSupported for agent with id %d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID));
#endif
    auto stay_matrix = FLAMEGPU->environment.getMacroProperty<unsigned int, TOTAL_AGENTS_ESTIMATION, SOLUTION_LENGTH>(STAY);

    const short currently_supported = FLAMEGPU->getVariable<short>(CURRENTLY_SUPPORTED);
    const short on_the_way_to_support = FLAMEGPU->getVariable<short>(ON_THE_WAY_TO_SUPPORT);
    const short requested_support = FLAMEGPU->getVariable<short>(REQUESTED_SUPPORT);
    const char requested_type = FLAMEGPU->getVariable<char>(REQUESTED_TYPE);
    const short requested_support_event_with_flow = FLAMEGPU->getVariable<short>(REQUESTED_SUPPORT_EVENT_WITH_FLOW);
    const short support_time_event = FLAMEGPU->getVariable<short>(SUPPORT_TIME_EVENT);
    const short contacts_id = FLAMEGPU->getVariable<short>(CONTACTS_ID);
    const unsigned char in_an_event = FLAMEGPU->getVariable<unsigned char>(IN_AN_EVENT);

    unsigned short next_index = FLAMEGPU->getVariable<unsigned short>(NEXT_INDEX);
    unsigned short target_index = FLAMEGPU->getVariable<unsigned short>(TARGET_INDEX);
    unsigned int stay = (unsigned int) stay_matrix[contacts_id][next_index];
    float agent_pos[3] = {FLAMEGPU->getVariable<float>(X), FLAMEGPU->getVariable<float>(Y), FLAMEGPU->getVariable<float>(Z)};

    if(requested_support != -1 && currently_supported != -1 && on_the_way_to_support == -1 &&
      (requested_support_event_with_flow == -1 || (in_an_event && requested_support_event_with_flow != -1) || support_time_event != -1)){
        // Send message with updated position to the support agent
        FLAMEGPU->message_out.setVariable<short>(CONTACTS_ID, NUMBER_OF_AGENTS_TYPES + contacts_id);
        FLAMEGPU->message_out.setVariable<float>(X, agent_pos[0]);
        FLAMEGPU->message_out.setVariable<float>(Y, agent_pos[1]);
        FLAMEGPU->message_out.setVariable<float>(Z, agent_pos[2]);

        if((next_index == target_index || (in_an_event == 1 && requested_support_event_with_flow == -1)) && ((stay == 1 && requested_type == ACCOMPANIMENT_AND_STAY) || requested_type == ACCOMPANIMENT_ONLY)){
            // The support is finished
            FLAMEGPU->message_out.setVariable<int>(REQUEST_ID, -2);

            FLAMEGPU->setVariable<short>(REQUESTED_SUPPORT, -1);
            FLAMEGPU->setVariable<char>(REQUESTED_TYPE, -1);
            FLAMEGPU->setVariable<short>(CURRENTLY_SUPPORTED, -1);
        }
        else{
            // The support continues
            FLAMEGPU->message_out.setVariable<int>(REQUEST_ID, -1);
        }

        // Event with support during the supported determined flow
        if(support_time_event){
            FLAMEGPU->message_out.setVariable<int>(SUPPORT_TIME, support_time_event);
            FLAMEGPU->setVariable<short>(SUPPORT_TIME_EVENT, -1);
        }
        else{
            FLAMEGPU->message_out.setVariable<int>(SUPPORT_TIME, -1);
        }

        FLAMEGPU->message_out.setKey(currently_supported);
    }

#if defined(DEBUG) && !defined(ENSEMBLE)
    printf("5,%d,%d,Ending beingSupported for agent with id %d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID));
#endif
    return ALIVE;
}


/**
    supportAgent

    Condition: initCondition

    Support an agent, if necessary.
*/
FLAMEGPU_AGENT_FUNCTION(supportAgent, MessageBucket, MessageNone) {
#if defined(DEBUG) && !defined(ENSEMBLE)
    printf("5,%d,%d,Starting supportAgent for agent with id %d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID));
#endif
    auto stay_matrix = FLAMEGPU->environment.getMacroProperty<unsigned int, TOTAL_AGENTS_ESTIMATION, SOLUTION_LENGTH>(STAY);
    auto coord2index = FLAMEGPU->environment.getMacroProperty<short, FLOORS, ENV_DIM_Z, ENV_DIM_X>(COORD2INDEX);

    const short currently_supported = FLAMEGPU->getVariable<short>(CURRENTLY_SUPPORTED);
    const short on_the_way_to_support = FLAMEGPU->getVariable<short>(ON_THE_WAY_TO_SUPPORT);
    const short requested_support = FLAMEGPU->getVariable<short>(REQUESTED_SUPPORT);
    const short agent_type = FLAMEGPU->getVariable<short>(AGENT_TYPE);
    const short contacts_id = FLAMEGPU->getVariable<short>(CONTACTS_ID);
    const unsigned short next_index = FLAMEGPU->getVariable<unsigned short>(NEXT_INDEX);
    const unsigned char in_an_event = FLAMEGPU->getVariable<unsigned char>(IN_AN_EVENT);

    unsigned short target_index = FLAMEGPU->getVariable<unsigned short>(TARGET_INDEX);

    // Support agent is supporting (update its position based on the received message)
    if(on_the_way_to_support == -1 && requested_support == -1 && currently_supported != -1){
        auto messages = FLAMEGPU->message_in(NUMBER_OF_AGENTS_TYPES + contacts_id);
        auto interested_message = messages.begin();

        const float agent_pos[3] = {FLAMEGPU->getVariable<float>(X), FLAMEGPU->getVariable<float>(Y), FLAMEGPU->getVariable<float>(Z)};

        if(interested_message != messages.end()){
            FLAMEGPU->setVariable<float>(X, (*interested_message).getVariable<float>(X));
            FLAMEGPU->setVariable<float>(Y, (*interested_message).getVariable<float>(Y));
            FLAMEGPU->setVariable<float>(Z, (*interested_message).getVariable<float>(Z));

            if(agent_pos[0] != (*interested_message).getVariable<float>(X) || agent_pos[1] != (*interested_message).getVariable<float>(Y) || agent_pos[2] != (*interested_message).getVariable<float>(Z))
                printf("0,%d,%d,%d,%d,%f,%f,%f,%d,%d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), contacts_id, agent_type, (*interested_message).getVariable<float>(X), (*interested_message).getVariable<float>(Y), (*interested_message).getVariable<float>(Z), FLAMEGPU->getVariable<unsigned char>(DISEASE_STATE), (short) coord2index[(unsigned short)((*interested_message).getVariable<float>(Y)/YOFFSET)][(unsigned short)(*interested_message).getVariable<float>(Z)][(unsigned short)(*interested_message).getVariable<float>(X)]);

            if((*interested_message).getVariable<int>(SUPPORT_TIME) != -1){
                int final_stay = (unsigned int) stay_matrix[contacts_id][target_index] - (*interested_message).getVariable<int>(SUPPORT_TIME);

                final_stay = final_stay > 0 ? final_stay: 1;

                stay_matrix[contacts_id][target_index].exchange(final_stay);
            }

            if((*interested_message).getVariable<int>(REQUEST_ID) == -2){
                FLAMEGPU->setVariable<short>(CURRENTLY_SUPPORTED, -1);
            }
        }
    }

#if defined(DEBUG) && !defined(ENSEMBLE)
    printf("5,%d,%d,Ending supportAgent for agent with id %d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID));
#endif
    return ALIVE;
}


/**
    outputPedestrianLocation

    Condition: initCondition

    Each pedestrian agent output a MessageSpatial3D message for counting contacts
*/
FLAMEGPU_AGENT_FUNCTION(outputPedestrianLocation, MessageNone, MessageSpatial3D) {
#if defined(DEBUG) && !defined(ENSEMBLE)
    printf("5,%d,%d,Beginning outputPedestrianLocation for agent with id %d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID));
#endif
    if (FLAMEGPU->getVariable<char>(SKIP_FLOW)) {
        return ALIVE;
    }

    auto coord2index = FLAMEGPU->environment.getMacroProperty<short, FLOORS, ENV_DIM_Z, ENV_DIM_X>(COORD2INDEX);

    const float agent_pos[3] = {FLAMEGPU->getVariable<float>(X), FLAMEGPU->getVariable<float>(Y), FLAMEGPU->getVariable<float>(Z)};
    const short node = coord2index[(unsigned short)(agent_pos[1]/YOFFSET)][(unsigned short)agent_pos[2]][(unsigned short)agent_pos[0]];

    FLAMEGPU->message_out.setVariable<id_t>(ID, FLAMEGPU->getID());
    FLAMEGPU->message_out.setVariable<short>(CONTACTS_ID, FLAMEGPU->getVariable<short>(CONTACTS_ID));
    FLAMEGPU->message_out.setVariable<unsigned char>(DISEASE_STATE, FLAMEGPU->getVariable<unsigned char>(DISEASE_STATE));
    FLAMEGPU->message_out.setVariable<short>(AGENT_TYPE, FLAMEGPU->getVariable<short>(AGENT_TYPE));
    FLAMEGPU->message_out.setVariable<short>(GRAPH_NODE, node);
    FLAMEGPU->message_out.setLocation(
        FLAMEGPU->getVariable<float>(X),
        FLAMEGPU->getVariable<float>(Y),
        FLAMEGPU->getVariable<float>(Z)
    );

#if defined(DEBUG) && !defined(ENSEMBLE)
    printf("5,%d,%d,Ending outputPedestrianLocation for agent with id %d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID));
#endif
    return ALIVE;
}


/** 
    printMoveAgentInfo

    Condition: initCondition

    Each pedestrian agent output its position and other info if it moved (for animation)
*/
FLAMEGPU_AGENT_FUNCTION(printMoveAgentInfo, MessageNone, MessageNone) {
#if defined(DEBUG) && !defined(ENSEMBLE)
    printf("5,%d,%d,Beginning printMoveAgentInfo for agent with id %d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID));
#endif
    const float agent_pos[3] = {FLAMEGPU->getVariable<float>(X), FLAMEGPU->getVariable<float>(Y), FLAMEGPU->getVariable<float>(Z)};

    auto coord2index = FLAMEGPU->environment.getMacroProperty<short, FLOORS, ENV_DIM_Z, ENV_DIM_X>(COORD2INDEX);

    float prev_x = FLAMEGPU->getVariable<float>(X_PREV);
    float prev_y = FLAMEGPU->getVariable<float>(Y_PREV);
    float prev_z = FLAMEGPU->getVariable<float>(Z_PREV);
    float agent_vel[3] = {FLAMEGPU->getVariable<float>(VELX), FLAMEGPU->getVariable<float>(VELY), FLAMEGPU->getVariable<float>(VELZ)};

    // Update animation
    if((agent_vel[0] != 0.0f || agent_vel[2] != 0.0f) && agent_vel[1] == 0.0f){
        float agent_animate = FLAMEGPU->getVariable<float>(ANIMATE) + (float) FLAMEGPU->getVariable<char>(ANIMATE_DIR);
        if (agent_animate >= 1.0f){
            agent_animate = 1.0f;
            FLAMEGPU->setVariable<char>(ANIMATE_DIR, -1);
        }
        else if (agent_animate <= 0.0f){
            agent_animate = 0.0f;
            FLAMEGPU->setVariable<char>(ANIMATE_DIR, 1);
        }
        FLAMEGPU->setVariable<float>(ANIMATE, agent_animate);
    }

    if(!compare_double(agent_pos[0], prev_x, 1e-10) || !compare_double(agent_pos[1], prev_y, 1e-10) || !compare_double(agent_pos[2], prev_z, 1e-10)){
        printf("0,%d,%d,%d,%d,%f,%f,%f,%d,%d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID), FLAMEGPU->getVariable<short>(AGENT_TYPE), agent_pos[0], agent_pos[1], agent_pos[2], FLAMEGPU->getVariable<unsigned char>(DISEASE_STATE), (short) coord2index[(unsigned short)(agent_pos[1]/YOFFSET)][(unsigned short)agent_pos[2]][(unsigned short)agent_pos[0]]);
    }

#if defined(DEBUG) && !defined(ENSEMBLE)
    printf("5,%d,%d,Ending printMoveAgentInfo for agent with id %d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID));
#endif
    return ALIVE;
}


/**
    outputPedestrianLocationAerosol

    Condition: initCondition

    Each pedestrian agent output a MessageBucket message for counting how many agents there are in a room (for aerosol transmission)
*/
FLAMEGPU_AGENT_FUNCTION(outputPedestrianLocationAerosol, MessageNone, MessageBucket) {
#if defined(DEBUG) && !defined(ENSEMBLE)
    printf("5,%d,%d,Beginning outputPedestrianLocationAerosol for agent with id %d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID));
#endif
    auto coord2index = FLAMEGPU->environment.getMacroProperty<short, FLOORS, ENV_DIM_Z, ENV_DIM_X>(COORD2INDEX);
    auto env_activity_type = FLAMEGPU->environment.getMacroProperty<float, NUMBER_OF_AGENTS_TYPES, NUMBER_OF_AGENTS_SUBTYPES, DAYS_IN_A_WEEK, FLOW_LENGTH>(ENV_ACTIVITY_TYPE);
    auto env_events_activity_type = FLAMEGPU->environment.getMacroProperty<float, NUMBER_OF_AGENTS_TYPES, EVENT_LENGTH>(ENV_EVENTS_ACTIVITY_TYPE);

    //qua sarà da considerare anche se è nella waiting room. Di certo non è attività pesante
    const float agent_pos[3] = {FLAMEGPU->getVariable<float>(X), FLAMEGPU->getVariable<float>(Y), FLAMEGPU->getVariable<float>(Z)};
    const short agent_type = FLAMEGPU->getVariable<short>(AGENT_TYPE);
    const short agent_subtype = FLAMEGPU->getVariable<short>(AGENT_SUBTYPE);
    const unsigned char waiting_room_flag = FLAMEGPU->getVariable<unsigned char>(WAITING_ROOM_FLAG);
    const short node = coord2index[(unsigned short)(agent_pos[1]/YOFFSET)][(unsigned short)agent_pos[2]][(unsigned short)agent_pos[0]];
    const unsigned short flow_index = FLAMEGPU->getVariable<unsigned short>(FLOW_INDEX);
    const unsigned char quarantine = FLAMEGPU->getVariable<unsigned char>(QUARANTINE);
    const unsigned char week_day_flow = FLAMEGPU->getVariable<unsigned char>(WEEK_DAY_FLOW);
    const unsigned char in_an_event = FLAMEGPU->getVariable<unsigned char>(IN_AN_EVENT);
    const short currently_supported = FLAMEGPU->getVariable<short>(CURRENTLY_SUPPORTED);
    const short on_the_way_to_support = FLAMEGPU->getVariable<short>(ON_THE_WAY_TO_SUPPORT);
    const short requested_support = FLAMEGPU->getVariable<short>(REQUESTED_SUPPORT);

    short event_id = FLAMEGPU->getVariable<short>(EVENT_ID);

    float activity_type = VERY_LIGHT_ACTIVITY;
    if(!quarantine && waiting_room_flag == OUTSIDE_WAITING_ROOM && requested_support != -1){
        activity_type = (float) env_activity_type[agent_type][agent_subtype][week_day_flow][flow_index];

        if(in_an_event == 1)
            activity_type = (float) env_events_activity_type[agent_type][event_id];
    }

    FLAMEGPU->message_out.setVariable<unsigned char>(DISEASE_STATE, FLAMEGPU->getVariable<unsigned char>(DISEASE_STATE));
    FLAMEGPU->message_out.setVariable<unsigned char>(MASK_TYPE, FLAMEGPU->getVariable<unsigned char>(MASK_TYPE));
    FLAMEGPU->message_out.setVariable<float>(ACTIVITY_TYPE, activity_type);

    FLAMEGPU->message_out.setKey(node);

#if defined(DEBUG) && !defined(ENSEMBLE)
    printf("5,%d,%d,Ending outputPedestrianLocationAerosol for agent with id %d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID));
#endif
    return ALIVE;
}


/**
    initAndNotFillingroomCondition

    Execute a function if the room is enabled and if the room is not a fillingroom
*/
FLAMEGPU_AGENT_FUNCTION_CONDITION(initAndNotFillingroomCondition) {
    auto coord2index = FLAMEGPU->environment.getMacroProperty<short, FLOORS, ENV_DIM_Z, ENV_DIM_X>(COORD2INDEX);

    unsigned short room_pos[3] = {FLAMEGPU->getVariable<unsigned short>(X_CENTER), FLAMEGPU->getVariable<unsigned short>(Y_CENTER), FLAMEGPU->getVariable<unsigned short>(Z_CENTER)};

    const short node = coord2index[(unsigned short)(room_pos[1]/YOFFSET)][(unsigned short)room_pos[2]][(unsigned short)room_pos[0]];
    const short node_type = FLAMEGPU->environment.getProperty<short, V>(NODE_TYPE, node);

    return FLAMEGPU->getVariable<unsigned char>(INIT_ROOM) && node_type != FILLINGROOM;
}

/**
    updateQuantaConcentration

    Condition: initAndNotFillingroomCondition

    Use the number of agents inside a room to update the quanta concentration of that room (for aerosol transmission)
*/
FLAMEGPU_AGENT_FUNCTION(updateQuantaConcentration, MessageBucket, MessageNone) {
#if defined(DEBUG) && !defined(ENSEMBLE)
    printf("5,%d,%d,Beginning updateQuantaConcentration for room with id %d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getID());
#endif
    unsigned short room_pos[3] = {FLAMEGPU->getVariable<unsigned short>(X_CENTER), FLAMEGPU->getVariable<unsigned short>(Y_CENTER), FLAMEGPU->getVariable<unsigned short>(Z_CENTER)};

    auto coord2index = FLAMEGPU->environment.getMacroProperty<short, FLOORS, ENV_DIM_Z, ENV_DIM_X>(COORD2INDEX);
    auto rooms_quanta_concentration = FLAMEGPU->environment.getMacroProperty<float, V>(ROOMS_QUANTA_CONCENTRATION);
    auto env_ventilation = FLAMEGPU->environment.getMacroProperty<float, DAYS, NUM_AREAS, NUM_ROOMS_TYPES>(ENV_VENTILATION);
    auto env_sterilisation = FLAMEGPU->environment.getMacroProperty<float, DAYS, NUM_AREAS, NUM_ROOMS_TYPES>(ENV_STERILISATION);
    auto env_air = FLAMEGPU->environment.getMacroProperty<float, DAYS, NUM_AREAS, NUM_ROOMS_TYPES>(ENV_AIR);

    const unsigned short day = FLAMEGPU->environment.getProperty<unsigned short>(DAY);
    const short area = FLAMEGPU->getVariable<short>(AREA);
    const short type = FLAMEGPU->getVariable<short>(TYPE);
    const short node = coord2index[(unsigned short)(room_pos[1]/YOFFSET)][(unsigned short)room_pos[2]][(unsigned short)room_pos[0]];
    const short node_type = FLAMEGPU->environment.getProperty<short, V>(NODE_TYPE, node);
    const float volume = FLAMEGPU->getVariable<float>(VOLUME);
    const float ventilation = (float) env_ventilation[day-1][area][type];
    const float sterilisation = (float) env_sterilisation[day-1][area][type];
    const float air = (float) env_air[day-1][area][type];
    const float vl = FLAMEGPU->environment.getProperty<float>(VL);
    const float ngen_base = FLAMEGPU->environment.getProperty<float>(NGEN_BASE);
    const float virus_variant_factor = FLAMEGPU->environment.getProperty<float>(VIRUS_VARIANT_FACTOR);
    const float gravitational_settling_rate = FLAMEGPU->environment.getProperty<float>(GRAVITATIONAL_SETTLING_RATE);
    const float decay_rate = FLAMEGPU->environment.getProperty<float>(DECAY_RATE);

    float total_n_r = 0.0f;
    for(const auto& message: FLAMEGPU->message_in(node)) {
        if(message.getVariable<unsigned char>(DISEASE_STATE) == INFECTED && node_type != CPOINT && node_type != DOOR){
            const float activity_type = message.getVariable<float>(ACTIVITY_TYPE);

            float exhalation_mask_efficacy = FLAMEGPU->environment.getProperty<float, 3>(EXHALATION_MASK_EFFICACY, message.getVariable<unsigned char>(MASK_TYPE));
            float base_n_r = ((activity_type * ngen_base) / pow(10, 9)) * virus_variant_factor;

            total_n_r += (base_n_r * pow(10, vl)) * (1 - exhalation_mask_efficacy);
        }
    }

    float total_first_order_lost_rate = gravitational_settling_rate + decay_rate + ventilation * (air + (1 - air) * sterilisation);

    float new_concentration = ((total_n_r / volume) / total_first_order_lost_rate) + (((float) rooms_quanta_concentration[node]) - ((total_n_r / volume) / total_first_order_lost_rate)) * exp(-(total_first_order_lost_rate * STEP));

    rooms_quanta_concentration[node].exchange(new_concentration);

    if(!((FLAMEGPU->getStepCounter() + START_STEP_TIME) % STEPS_IN_A_HOUR)){
        if(!compare_double((float) new_concentration, 0.0f, 1e-10f))
            printf("3,%d,%d,%.10f,%d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), (float) new_concentration, node);

        FLAMEGPU->setVariable<float>(QUANTA_CONCENTRATION, FLAMEGPU->getVariable<float>(QUANTA_CONCENTRATION) + new_concentration);
    }

#if defined(DEBUG) && !defined(ENSEMBLE)
    printf("5,%d,%d,Ending updateQuantaConcentration for room with id %d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getID());
#endif
    return ALIVE;
}


/**
    notInitAndNotFillingroomCondition

    Execute a function if the room is not enabled and if the room is not a fillingroom
*/
FLAMEGPU_AGENT_FUNCTION_CONDITION(notInitAndNotFillingroomCondition) {
    auto coord2index = FLAMEGPU->environment.getMacroProperty<short, FLOORS, ENV_DIM_Z, ENV_DIM_X>(COORD2INDEX);

    unsigned short room_pos[3] = {FLAMEGPU->getVariable<unsigned short>(X_CENTER), FLAMEGPU->getVariable<unsigned short>(Y_CENTER), FLAMEGPU->getVariable<unsigned short>(Z_CENTER)};

    const short node = coord2index[(unsigned short)(room_pos[1]/YOFFSET)][(unsigned short)room_pos[2]][(unsigned short)room_pos[0]];
    const short node_type = FLAMEGPU->environment.getProperty<short, V>(NODE_TYPE, node);

    return !FLAMEGPU->getVariable<unsigned char>(INIT_ROOM) && node_type != FILLINGROOM;
}

/**
    outputRoomLocation

    Condition: notInitAndNotFillingroomCondition

    Each room agent output a MessageBucket message with its position(for handling events)
*/
FLAMEGPU_AGENT_FUNCTION(outputRoomLocation, MessageNone, MessageBucket) {
#if defined(DEBUG) && !defined(ENSEMBLE)
    printf("5,%d,%d,Begin outputRoomLocation for room with id %d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getID());
#endif
    // Initialize curand
    auto cuda_rng_offsets_room = FLAMEGPU->environment.getMacroProperty<unsigned int, NUM_ROOMS>(CUDA_RNG_OFFSETS_ROOM);
    auto coord2index = FLAMEGPU->environment.getMacroProperty<short, FLOORS, ENV_DIM_Z, ENV_DIM_X>(COORD2INDEX);

    curand_init(FLAMEGPU->environment.getProperty<unsigned int>(SEED), TOTAL_AGENTS_ESTIMATION+FLAMEGPU->getID(), cuda_rng_offsets_room[FLAMEGPU->getID()-1], &cuda_room_states[FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX)][FLAMEGPU->getID()-1]);

    unsigned short room_pos[3] = {FLAMEGPU->getVariable<unsigned short>(X_CENTER), FLAMEGPU->getVariable<unsigned short>(Y_CENTER), FLAMEGPU->getVariable<unsigned short>(Z_CENTER)};

    const short node = (short) coord2index[(unsigned short)(room_pos[1]/YOFFSET)][(unsigned short)room_pos[2]][(unsigned short)room_pos[0]];

    FLAMEGPU->message_out.setVariable<unsigned short>(X, room_pos[0]);
    FLAMEGPU->message_out.setVariable<unsigned short>(Y, room_pos[1]);
    FLAMEGPU->message_out.setVariable<unsigned short>(Z, room_pos[2]);
    FLAMEGPU->message_out.setVariable<short>(GRAPH_NODE, node);
    FLAMEGPU->message_out.setVariable<short>(AREA, FLAMEGPU->getVariable<short>(AREA));

    FLAMEGPU->message_out.setKey(FLAMEGPU->environment.getProperty<short, V>(NODE_TYPE, node));

    FLAMEGPU->setVariable<unsigned char>(INIT_ROOM, 1);

#if defined(DEBUG) && !defined(ENSEMBLE)
    printf("5,%d,%d,Ending outputRoomLocation for room with id %d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getID());
#endif
    return ALIVE;
}


/**
    updateQuantaInhaledAndContacts

    Condition: initCondition

    Use the quanta concentration in the room to update the quanta inhaled by the agent (for aerosol transmission) and count contacts
*/
FLAMEGPU_AGENT_FUNCTION(updateQuantaInhaledAndContacts, MessageSpatial3D, MessageNone) {
#if defined(DEBUG) && !defined(ENSEMBLE)
    printf("5,%d,%d,Beginning updateQuantaInhaledAndContacts for room with id %d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID));
#endif
    // Update quanta inhaled
    auto coord2index = FLAMEGPU->environment.getMacroProperty<short, FLOORS, ENV_DIM_Z, ENV_DIM_X>(COORD2INDEX);

    const float agent_pos[3] = {FLAMEGPU->getVariable<float>(X), FLAMEGPU->getVariable<float>(Y), FLAMEGPU->getVariable<float>(Z)};
    const short node = coord2index[(unsigned short)(agent_pos[1]/YOFFSET)][(unsigned short)agent_pos[2]][(unsigned short)agent_pos[0]];

    if(FLAMEGPU->getVariable<unsigned char>(DISEASE_STATE) == SUSCEPTIBLE){
        auto rooms_quanta_concentration = FLAMEGPU->environment.getMacroProperty<float, V>(ROOMS_QUANTA_CONCENTRATION);
        auto env_activity_type = FLAMEGPU->environment.getMacroProperty<float, NUMBER_OF_AGENTS_TYPES, NUMBER_OF_AGENTS_SUBTYPES, DAYS_IN_A_WEEK, FLOW_LENGTH>(ENV_ACTIVITY_TYPE);
        auto env_events_activity_type = FLAMEGPU->environment.getMacroProperty<float, NUMBER_OF_AGENTS_TYPES, EVENT_LENGTH>(ENV_EVENTS_ACTIVITY_TYPE);

        const short agent_type = FLAMEGPU->getVariable<short>(AGENT_TYPE);
        const short agent_subtype = FLAMEGPU->getVariable<short>(AGENT_SUBTYPE);
        const unsigned char mask_type = FLAMEGPU->getVariable<unsigned char>(MASK_TYPE);
        const float inhalation_rate_pure = FLAMEGPU->environment.getProperty<float>(INHALATION_RATE_PURE);
        const float inhalation_mask_efficacy = FLAMEGPU->environment.getProperty<float, 3>(INHALATION_MASK_EFFICACY, mask_type);
        const unsigned short flow_index = FLAMEGPU->getVariable<unsigned short>(FLOW_INDEX);
        const unsigned char quarantine = FLAMEGPU->getVariable<unsigned char>(QUARANTINE);
        const unsigned char week_day_flow = FLAMEGPU->getVariable<unsigned char>(WEEK_DAY_FLOW);
        const unsigned char in_an_event = FLAMEGPU->getVariable<unsigned char>(IN_AN_EVENT);

        short event_id = FLAMEGPU->getVariable<short>(EVENT_ID);
        float activity_type = VERY_LIGHT_ACTIVITY;
        if(!quarantine && flow_index > 0){
            activity_type = (float) env_activity_type[agent_type][agent_subtype][week_day_flow][flow_index];

            if(in_an_event == 1)
                activity_type = (float) env_events_activity_type[agent_type][event_id];
        }

        float inhalation_rate = (inhalation_rate_pure * (1 - inhalation_mask_efficacy) * activity_type) / 1000;
        float concentration = (float) (node != -1) ? rooms_quanta_concentration[node]: 0.0f;

        float quanta_inhaled = FLAMEGPU->getVariable<float>(QUANTA_INHALED);
        quanta_inhaled = quanta_inhaled + inhalation_rate * STEP * concentration;
        FLAMEGPU->setVariable<float>(QUANTA_INHALED, quanta_inhaled);
    }

    // Update contacts
    const short contacts_id = FLAMEGPU->getVariable<short>(CONTACTS_ID);
    const short agent_type = FLAMEGPU->getVariable<short>(AGENT_TYPE);

    for (const auto& message: FLAMEGPU->message_in(agent_pos[0], agent_pos[1], agent_pos[2])) {
        const float near_agent_pos[3] = {message.getVariable<float>(X), message.getVariable<float>(Y), message.getVariable<float>(Z)};
        const short message_contacts_id = message.getVariable<short>(CONTACTS_ID);
        const short message_agent_type = message.getVariable<short>(AGENT_TYPE);

        float x_diff = near_agent_pos[0] - agent_pos[0];
        float y_diff = near_agent_pos[1] - agent_pos[1];
        float z_diff = near_agent_pos[2] - agent_pos[2];
        const float separation = sqrt(x_diff*x_diff + y_diff*y_diff + z_diff*z_diff);

        if (separation > 0.0f && separation < RADIUS && node == message.getVariable<short>(GRAPH_NODE)){
            // Count contact inside the MacroProperty (each step)
            auto contacts_matrix = FLAMEGPU->environment.getMacroProperty<unsigned int, NUMBER_OF_AGENTS_TYPES_PLUS_1, NUMBER_OF_AGENTS_TYPES_PLUS_1>(CONTACTS_MATRIX);
            contacts_matrix[agent_type][message_agent_type]++;

            // Count contacts among a susceptible agent and an infected one.
            if(FLAMEGPU->getVariable<unsigned char>(DISEASE_STATE) == SUSCEPTIBLE && message.getVariable<unsigned char>(DISEASE_STATE) == INFECTED){
                unsigned char infected_contact = FLAMEGPU->getVariable<unsigned char>(INFECTED_CONTACT);
                FLAMEGPU->setVariable<unsigned char>(INFECTED_CONTACT, infected_contact + 1);
                printf("1,%d,%d,%d,%d,%d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), agent_type, message.getVariable<short>(AGENT_TYPE), (short) coord2index[(unsigned short)(agent_pos[1]/YOFFSET)][(unsigned short)agent_pos[2]][(unsigned short)agent_pos[0]]);
            }
        }
    }

#if defined(DEBUG) && !defined(ENSEMBLE)
    printf("5,%d,%d,Ending updateQuantaInhaledAndContacts for room with id %d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID));
#endif
    return ALIVE;
}


/** 
    waitingInWaitingRoom

    Condition: initCondition

    When the agent is waiting inside a waiting room he checks if the room has notificated that it's free and he can go.
    Otherwise, he waits.
*/

FLAMEGPU_AGENT_FUNCTION(waitingInWaitingRoom, MessageBucket, MessageBucket) {
#if defined(DEBUG) && !defined(ENSEMBLE)
    printf("5,%d,%d,Beginning waitingInWaitingRoom for agent with id %d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID));
#endif
    if(FLAMEGPU->getVariable<unsigned char>(WAITING_ROOM_FLAG) == INSIDE_WAITING_ROOM){
        auto stay_matrix = FLAMEGPU->environment.getMacroProperty<unsigned int, TOTAL_AGENTS_ESTIMATION, SOLUTION_LENGTH>(STAY);

        const short contacts_id = FLAMEGPU->getVariable<short>(CONTACTS_ID);

        //if the agent is not already exiting from waiting room
        if(FLAMEGPU->getVariable<unsigned char>(ENTRY_EXIT_FLAG) == STAYING_IN_WAITING_ROOM){
            bool free = false;

            for(const auto& message: FLAMEGPU->message_in(contacts_id)){
                free = true;
                break;
            }

            // If the room he's waiting for it's not free
            if(!free) {
                unsigned short target_index = FLAMEGPU->getVariable<unsigned short>(TARGET_INDEX);
                short time_waiting = FLAMEGPU->getVariable<short>(WAITING_ROOM_TIME);

                ++time_waiting;

                FLAMEGPU->setVariable<short>(WAITING_ROOM_TIME, time_waiting);
                stay_matrix[contacts_id][target_index].exchange(2);

                FLAMEGPU->message_out.setVariable<short>(CONTACTS_ID, FLAMEGPU->getVariable<short>(CONTACTS_ID));
                FLAMEGPU->message_out.setVariable<short>(AGENT_TYPE, FLAMEGPU->getVariable<short>(AGENT_TYPE));
                FLAMEGPU->message_out.setVariable<short>(WAITING_ROOM_TIME, time_waiting);

                FLAMEGPU->message_out.setKey(FLAMEGPU->getVariable<short>(NODE_WAITING_FOR));
            }
            // If the room he's waiting for it's free
            else {
                unsigned short flow_index = FLAMEGPU->getVariable<unsigned short>(FLOW_INDEX) - 1;
                unsigned short target_index = FLAMEGPU->getVariable<unsigned short>(TARGET_INDEX);

                FLAMEGPU->setVariable<unsigned short>(FLOW_INDEX, flow_index);
                FLAMEGPU->setVariable<unsigned char>(ENTRY_EXIT_FLAG, EXITING_FROM_WAITING_ROOM);
                stay_matrix[contacts_id][target_index].exchange(1);
            }
        }
    }

#if defined(DEBUG) && !defined(ENSEMBLE)
    printf("5,%d,%d,Ending waitingInWaitingRoom for agent with id %d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID));
#endif
    return ALIVE;
}


/**
    handlingQueueinWaitingRoom

    Condition: -

    The room monitors the agent waiting inside it, if it's a waiting room. If it's free notifies it otherwise waits
*/
FLAMEGPU_AGENT_FUNCTION(handlingQueueinWaitingRoom, MessageBucket, MessageBucket) {
#if defined(DEBUG) && !defined(ENSEMBLE)
    printf("5,%d,%d,Beginning handlingQueueInWaitingRoom for room with id %d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getID());
#endif
    auto coord2index = FLAMEGPU->environment.getMacroProperty<short, FLOORS, ENV_DIM_Z, ENV_DIM_X>(COORD2INDEX);
    auto global_resources = FLAMEGPU->environment.getMacroProperty<unsigned int, V>(GLOBAL_RESOURCES);
    auto global_resources_counter = FLAMEGPU->environment.getMacroProperty<unsigned int, V>(GLOBAL_RESOURCES_COUNTER);
    auto specific_resources = FLAMEGPU->environment.getMacroProperty<unsigned int, NUMBER_OF_AGENTS_TYPES, V>(SPECIFIC_RESOURCES);
    auto specific_resources_counter = FLAMEGPU->environment.getMacroProperty<unsigned int, NUMBER_OF_AGENTS_TYPES, V>(SPECIFIC_RESOURCES_COUNTER);

    unsigned short room_pos[3] = {FLAMEGPU->getVariable<unsigned short>(X_CENTER), FLAMEGPU->getVariable<unsigned short>(Y_CENTER), FLAMEGPU->getVariable<unsigned short>(Z_CENTER)};

    const short node = coord2index[(unsigned short)(room_pos[1]/YOFFSET)][(unsigned short)room_pos[2]][(unsigned short)room_pos[0]];

    for(const auto& message: FLAMEGPU->message_in(node)){
        short agent_type = message.getVariable<short>(AGENT_TYPE);
        unsigned int get_specific_resource = ++specific_resources_counter[agent_type][node];

        if(get_specific_resource <= specific_resources[agent_type][node]){
            unsigned int get_global_resource = ++global_resources_counter[node];

            if(get_global_resource <= global_resources[node]){
                int max_time_waiting = INT_MIN;
                short contacts_id = -1;

                for(const auto& message: FLAMEGPU->message_in(node)){
                    if(message.getVariable<short>(WAITING_ROOM_TIME) > max_time_waiting && message.getVariable<short>(AGENT_TYPE) == agent_type){
                        max_time_waiting = message.getVariable<short>(WAITING_ROOM_TIME);
                        contacts_id = message.getVariable<short>(CONTACTS_ID);
                    }
                }
                FLAMEGPU->message_out.setVariable<short>(GRAPH_NODE, node);
                FLAMEGPU->message_out.setKey(contacts_id);
                break;
            }
            else {
                --global_resources_counter[node];
                --specific_resources_counter[agent_type][node];
            }

        }
        else {
            --specific_resources_counter[agent_type][node];
        }
    }

#if defined(DEBUG) && !defined(ENSEMBLE)
    printf("5,%d,%d,Ending handlingQueueInWaitingRoom for room with id %d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getID());
#endif
    return ALIVE;
}

#endif //_AGENT_FUNCTIONS_CUH_