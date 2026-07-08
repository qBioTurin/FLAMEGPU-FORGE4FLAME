#ifndef _DEVICE_FUNCTIONS_CUH_
#define _DEVICE_FUNCTIONS_CUH_

#include "defines.h"

using namespace flamegpu;
using namespace std;

namespace device_functions {
    /**
     * Compare floating points.
    */
    FLAMEGPU_DEVICE_FUNCTION bool compare_double(const double a, const double b, const double epsilon) {
        return fabs(a - b) < epsilon;
    }

    /**
     * Generate a random number using the given RNG, distribution and parameters for pedestrians.
    */
    template<typename MessageIn, typename MessageOut>
    FLAMEGPU_DEVICE_FUNCTION double cuda_pedestrian_rng(DeviceAPI<MessageIn, MessageOut>* FLAMEGPU, unsigned short distribution_id, curandState *cuda_states, short type, short id, double a, double b, bool flow_time) {
        double random = (type == TRUNCATED_POSITIVE_NORMAL) ? curand_normal(&cuda_states[id]): curand_uniform(&cuda_states[id]);

        if(type == EXPONENTIAL && compare_double((double) random, 1.0f, 1e-10f)){
            do{
                random = curand_uniform(&cuda_states[id]);
            }while(compare_double((double) random, 1.0f, 1e-10f));
        }
        const double event_time_random = DISTRIBUTION(type, random, a, b);

        auto cuda_rng_offsets_pedestrian = FLAMEGPU->environment.template getMacroProperty<unsigned int, TOTAL_AGENTS_ESTIMATION>(CUDA_RNG_OFFSETS_PEDESTRIAN);
        cuda_rng_offsets_pedestrian[FLAMEGPU->template getVariable<short>(CONTACTS_ID)]++;

        return (flow_time && event_time_random < 1.0f) ? 1.0f: event_time_random;
    }

    /**
     * Generate a random number using the given RNG, distribution and parameters for rooms.
    */
    FLAMEGPU_DEVICE_FUNCTION double cuda_room_rng(DeviceAPI<MessageBucket, MessageBucket>* FLAMEGPU, unsigned short distribution_id, curandState *cuda_states, short type, short id, double a, double b, bool flow_time) {
        double random = (type == TRUNCATED_POSITIVE_NORMAL) ? curand_normal(&cuda_states[id]): curand_uniform(&cuda_states[id]);

        if(type == EXPONENTIAL && compare_double((double) random, 1.0f, 1e-10f)){
            do{
                random = curand_uniform(&cuda_states[id]);
            }while(compare_double((double) random, 1.0f, 1e-10f));
        }
        const double event_time_random = DISTRIBUTION(type, random, a, b);

        auto cuda_rng_offsets_room = FLAMEGPU->environment.template getMacroProperty<unsigned int, NUM_ROOMS>(CUDA_RNG_OFFSETS_ROOM);
        cuda_rng_offsets_room[FLAMEGPU->getID()]++;

        return (flow_time && event_time_random < 1.0f) ? 1.0f: event_time_random;
    }

    /**
     * Generate a random offset inside rooms.
    */
    template<typename MessageIn, typename MessageOut>
    FLAMEGPU_DEVICE_FUNCTION void generate_offset(DeviceAPI<MessageIn, MessageOut>* FLAMEGPU, float* jitter_x, float* jitter_z, short new_target){
#if defined(DEBUG) && !defined(ENSEMBLE)
        printf("5,%d,%d,Beginning of generate_offset for agent with id %d\n", FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->template getVariable<short>(CONTACTS_ID));
#endif
        const short contacts_id = FLAMEGPU->template getVariable<short>(CONTACTS_ID);
        const float yaw = FLAMEGPU->environment.template getProperty<float, V>(NODE_YAW, new_target);
        const bool yaw_condition = compare_double(yaw, M_PI/2, 0.5f) || compare_double(yaw, 2*M_PI - M_PI/2, 0.5f);

        float length = FLAMEGPU->environment.template getProperty<float, V>(NODE_LENGTH, new_target);
        float width = FLAMEGPU->environment.template getProperty<float, V>(NODE_WIDTH, new_target);
        float offset_x, offset_z;

        offset_x = yaw_condition ? width: length;
        offset_z = yaw_condition ? length: width;

        *jitter_x = cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_JITTER_X_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX)], UNIFORM, contacts_id, 0.0f, offset_x - 1e-3, false);
        *jitter_z = cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_JITTER_Z_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX)], UNIFORM, contacts_id, 0.0f, offset_z - 1e-3, false);
#if defined(DEBUG) && !defined(ENSEMBLE)
        printf("5,%d,%d,Ending of generate_offset for agent with id %d\n", FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->template getVariable<short>(CONTACTS_ID));
#endif
    }

 /**
     * Find a room of free resources for an event, searching the nearest
    */
    template<typename MessageOut>
    FLAMEGPU_DEVICE_FUNCTION short findFreeRoomForEventOfTypeAndArea(DeviceAPI<MessageBucket, MessageOut>* FLAMEGPU, float previous_separation, int type_room_event, int area_room_event, bool *available) {
#if defined(DEBUG) && !defined(ENSEMBLE)
        printf("5,%d,%d,Beginning of findFreeRoomForEventOfTypeAndArea for agent with id %d\n", FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->template getVariable<short>(CONTACTS_ID));
#endif
        const short agent_type = FLAMEGPU->template getVariable<short>(AGENT_TYPE);

        short event_node;
        float min_separation = numeric_limits<float>::max();
        float agent_pos[3] = {FLAMEGPU->template getVariable<float>(X), FLAMEGPU->template getVariable<float>(Y), FLAMEGPU->template getVariable<float>(Z)};

        auto global_resources = FLAMEGPU->environment.template getMacroProperty<unsigned int, V>(GLOBAL_RESOURCES);
        auto global_resources_counter = FLAMEGPU->environment.template getMacroProperty<unsigned int, V>(GLOBAL_RESOURCES_COUNTER);
        auto specific_resources = FLAMEGPU->environment.template getMacroProperty<unsigned int, NUMBER_OF_AGENTS_TYPES, V>(SPECIFIC_RESOURCES);
        auto specific_resources_counter = FLAMEGPU->environment.template getMacroProperty<unsigned int, NUMBER_OF_AGENTS_TYPES, V>(SPECIFIC_RESOURCES_COUNTER);

        do {
            // Searching the nearest room related to the event
            // To add the area
            float min_separation = numeric_limits<float>::max();
            event_node = -1;
            for(const auto& message: FLAMEGPU->message_in(type_room_event)) {
                const unsigned short near_agent_pos[3] = {message.template getVariable<unsigned short>(X), message.template getVariable<unsigned short>(Y), message.template getVariable<unsigned short>(Z)};

                short area_room = message.template getVariable<short>(AREA);

                float separation = abs(near_agent_pos[0] - agent_pos[0]) + abs(near_agent_pos[1] - agent_pos[1]) + abs(near_agent_pos[2] - agent_pos[2]);
                if(separation < min_separation && separation > previous_separation && area_room_event == area_room){
                    min_separation = separation;
                    event_node = message.template getVariable<short>(GRAPH_NODE);
                }

            }

            // Try getting the resources of the room
            if(event_node != -1){
                bool event_resources = false;
                int get_specific_resource = ++specific_resources_counter[agent_type][event_node];

                if(get_specific_resource <= specific_resources[agent_type][event_node]){
                    int get_global_resource = ++global_resources_counter[event_node];

                    if(get_global_resource <= global_resources[event_node]){
                        *available = true;
                        event_resources = true;
                    }
                    else {
                        --global_resources_counter[event_node];
                    }
                }

                if(!event_resources){
                    --specific_resources_counter[agent_type][event_node];
                    previous_separation = min_separation;
                }
            }
        }
        while(!*available && event_node != -1);

#if defined(DEBUG) && !defined(ENSEMBLE)
        printf("5,%d,%d,Ending of findFreeRoomForEventOfTypeAndAre for agent with id %d\n", FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->template getVariable<short>(CONTACTS_ID));
#endif
        return event_node;
    }

    /**
     * Find a room of free resources
    */
    template<typename MessageOut>
    FLAMEGPU_DEVICE_FUNCTION short findFreeRoomOfTypeAndArea(DeviceAPI<MessageBucket, MessageOut>* FLAMEGPU, int flow, int random, int lenght_rooms, unsigned short* ward_indeces, bool *available) {
#if defined(DEBUG) && !defined(ENSEMBLE)
        printf("5,%d,%d,Beginning of findFreeRoomOfTypeAndArea for agent with id %d\n", FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->template getVariable<short>(CONTACTS_ID));
#endif
        const short agent_type = FLAMEGPU->template getVariable<short>(AGENT_TYPE);
        const short contacts_id = FLAMEGPU->template getVariable<short>(CONTACTS_ID);

        int random_iterator = random;
        unsigned short final_target;

        auto global_resources = FLAMEGPU->environment.template getMacroProperty<unsigned int, V>(GLOBAL_RESOURCES);
        auto global_resources_counter = FLAMEGPU->environment.template getMacroProperty<unsigned int, V>(GLOBAL_RESOURCES_COUNTER);
        auto specific_resources = FLAMEGPU->environment.template getMacroProperty<unsigned int, NUMBER_OF_AGENTS_TYPES, V>(SPECIFIC_RESOURCES);
        auto specific_resources_counter = FLAMEGPU->environment.template getMacroProperty<unsigned int, NUMBER_OF_AGENTS_TYPES, V>(SPECIFIC_RESOURCES_COUNTER);

        auto messages = FLAMEGPU->message_in(flow);
        bool room_resources = false;
        do {
            auto list_front = messages.begin();

            for(int i = 0; i < ward_indeces[random_iterator]; i++) list_front++;

            final_target = (*list_front).template getVariable<short>(GRAPH_NODE);

            // Try getting the resources of the room
            unsigned int get_specific_resource = ++specific_resources_counter[agent_type][final_target];
            room_resources = false;

            if(get_specific_resource <= specific_resources[agent_type][final_target]){
                int get_global_resource = ++global_resources_counter[final_target];

                if(get_global_resource <= global_resources[final_target]){
                    *available = true;
                    room_resources = true;
                }
                else{
                    --global_resources_counter[final_target];
                }
            }

            if(!room_resources) {
                --specific_resources_counter[agent_type][final_target];
                random_iterator = (random_iterator + 1) % lenght_rooms;
            }
        }
        while(!*available && random_iterator != random);

#if defined(DEBUG) && !defined(ENSEMBLE)
        printf("5,%d,%d,Ending of findFreeRoomOfTypeAndArea for agent with id %d\n", FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->template getVariable<short>(CONTACTS_ID));
#endif
        return final_target;
    }

    /**
     * Take the next destination inside the determined flow of the agent.
    */
    template<typename MessageOut>
    FLAMEGPU_DEVICE_FUNCTION short take_new_destination_flow(DeviceAPI<MessageBucket, MessageOut> *FLAMEGPU, int *stay, const short start_node, bool *available, const bool identified = false, const unsigned short severity = MINOR){
#if defined(DEBUG) && !defined(ENSEMBLE)
        printf("5,%d,%d,Beginning of take_new_destination_flow for agent with id %d\n", FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->template getVariable<short>(CONTACTS_ID));
#endif
        unsigned short flow_index = FLAMEGPU->template getVariable<unsigned short>(FLOW_INDEX) + 1;
        unsigned char week_day_flow = FLAMEGPU->template getVariable<unsigned char>(WEEK_DAY_FLOW);
        short final_target;

        const short start_node_type = FLAMEGPU->environment.template getProperty<short, V>(NODE_TYPE, start_node);
        const unsigned short day = FLAMEGPU->environment.template getProperty<unsigned short>(DAY);
        const short agent_type = FLAMEGPU->template getVariable<short>(AGENT_TYPE);
        const short agent_subtype = FLAMEGPU->template getVariable<short>(AGENT_SUBTYPE);
        const short contacts_id = FLAMEGPU->template getVariable<short>(CONTACTS_ID);

        auto spawnrooms_areas_ids = FLAMEGPU->environment.template getMacroProperty<unsigned short, NUM_AREAS, NUM_SPAWNROOM + 1>(SPAWNROOMS_AREAS_IDS);
        auto env_flow = FLAMEGPU->environment.template getMacroProperty<short, NUMBER_OF_AGENTS_TYPES, NUMBER_OF_AGENTS_SUBTYPES, DAYS_IN_A_WEEK, FLOW_LENGTH>(ENV_FLOW);
        auto env_flow_distr = FLAMEGPU->environment.template getMacroProperty<short, NUMBER_OF_AGENTS_TYPES, NUMBER_OF_AGENTS_SUBTYPES, DAYS_IN_A_WEEK, FLOW_LENGTH>(ENV_FLOW_DISTR);
        auto env_flow_distr_firstparam = FLAMEGPU->environment.template getMacroProperty<int, NUMBER_OF_AGENTS_TYPES, NUMBER_OF_AGENTS_SUBTYPES, DAYS_IN_A_WEEK, FLOW_LENGTH>(ENV_FLOW_DISTR_FIRSTPARAM);
        auto env_flow_distr_secondparam = FLAMEGPU->environment.template getMacroProperty<int, NUMBER_OF_AGENTS_TYPES, NUMBER_OF_AGENTS_SUBTYPES, DAYS_IN_A_WEEK, FLOW_LENGTH>(ENV_FLOW_DISTR_SECONDPARAM);
        auto env_flow_area = FLAMEGPU->environment.template getMacroProperty<short, NUMBER_OF_AGENTS_TYPES, NUMBER_OF_AGENTS_SUBTYPES, DAYS_IN_A_WEEK, FLOW_LENGTH>(ENV_FLOW_AREA);
        auto env_room_for_quarantine_type = FLAMEGPU->environment.template getMacroProperty<short, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_ROOM_FOR_QUARANTINE_TYPE);
        auto env_room_for_quarantine_area = FLAMEGPU->environment.template getMacroProperty<short, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_ROOM_FOR_QUARANTINE_AREA);

        // Resources
        auto global_resources = FLAMEGPU->environment.template getMacroProperty<unsigned int, V>(GLOBAL_RESOURCES);
        auto global_resources_counter = FLAMEGPU->environment.template getMacroProperty<unsigned int, V>(GLOBAL_RESOURCES_COUNTER);
        auto specific_resources = FLAMEGPU->environment.template getMacroProperty<unsigned int, NUMBER_OF_AGENTS_TYPES, V>(SPECIFIC_RESOURCES);
        auto specific_resources_counter = FLAMEGPU->environment.template getMacroProperty<unsigned int, NUMBER_OF_AGENTS_TYPES, V>(SPECIFIC_RESOURCES_COUNTER);
        auto alternative_resources_area_det = FLAMEGPU->environment.template getMacroProperty<short, NUMBER_OF_AGENTS_TYPES, V>(ALTERNATIVE_RESOURCES_AREA_DET);
        auto alternative_resources_type_det = FLAMEGPU->environment.template getMacroProperty<short, NUMBER_OF_AGENTS_TYPES, V>(ALTERNATIVE_RESOURCES_TYPE_DET);

        const int flow = (short) env_flow[agent_type][agent_subtype][week_day_flow][flow_index];
        const int flow_area = (short) env_flow_area[agent_type][agent_subtype][week_day_flow][flow_index];

        FLAMEGPU->template setVariable<unsigned short>(FLOW_INDEX, flow_index);

        if((short) env_flow_distr[agent_type][agent_subtype][week_day_flow][flow_index] != -1)
            *stay = (unsigned int) cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_FLOW_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX)], (short) env_flow_distr[agent_type][agent_subtype][week_day_flow][flow_index], contacts_id, (float) env_flow_distr_firstparam[agent_type][agent_subtype][week_day_flow][flow_index], (float) env_flow_distr_secondparam[agent_type][agent_subtype][week_day_flow][flow_index], true);

        unsigned short j = 0;
        if((flow != SPAWNROOM && !identified) || (severity == MAJOR && (short) env_room_for_quarantine_type[day-1][agent_type] != SPAWNROOM)){
            auto messages = FLAMEGPU->message_in(flow);
            int area = flow_area;

            if(severity == MAJOR){
                messages = FLAMEGPU->message_in((short) env_room_for_quarantine_type[day-1][agent_type]);
                area = (short) env_room_for_quarantine_area[day-1][agent_type];
            }

            unsigned short ward_indeces[SOLUTION_LENGTH];

            auto i = messages.begin();
            unsigned short k = 0;
            while(i != messages.end()){
                const short local_area = (*i).template getVariable<short>(AREA);

                if(local_area == area){
                    ward_indeces[j] = k;
                    j++;
                }

                i++;
                k++;
            }

            int random = round(cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_TAKE_NEW_DESTINATION_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX)], UNIFORM, contacts_id, 0.0f, (float) (j-1), false));
            int random_iterator = random;
            int lenght_rooms = j;
            unsigned int get_global_resource;
            unsigned int get_specific_resource;

            // If the agent is already waiting for a node, go for it
            if(FLAMEGPU->template getVariable<short>(NODE_WAITING_FOR) != -1){
                final_target = FLAMEGPU->template getVariable<short>(NODE_WAITING_FOR);
                *available = true;
            }
            else {
                auto list_front = messages.begin();

                for(int i = 0; i < ward_indeces[random]; i++) list_front++;

                final_target = (*list_front).template getVariable<short>(GRAPH_NODE);
            }

            if(severity == MINOR){
                if((short) alternative_resources_type_det[agent_type][final_target] == WAITINGROOM && FLAMEGPU->template getVariable<unsigned char>(WAITING_ROOM_FLAG) == OUTSIDE_WAITING_ROOM){
                    *available = true;

                    float agent_pos[3] = {FLAMEGPU->template getVariable<float>(X), FLAMEGPU->template getVariable<float>(Y), FLAMEGPU->template getVariable<float>(Z)};
                    FLAMEGPU->template setVariable<short>(NODE_WAITING_FOR, final_target);
                    float min_separation = numeric_limits<float>::max();

                    for(const auto& message: FLAMEGPU->message_in(WAITINGROOM)) {
                        const unsigned short near_agent_pos[3] = {message.template getVariable<unsigned short>(X), message.template getVariable<unsigned short>(Y), message.template getVariable<unsigned short>(Z)};

                        float separation = abs(near_agent_pos[0] - agent_pos[0]) + abs(near_agent_pos[1] - agent_pos[1]) + abs(near_agent_pos[2] - agent_pos[2]);
                        if(separation < min_separation){
                            min_separation = separation;
                            final_target = message.template getVariable<short>(GRAPH_NODE);
                        }
                    }

                    FLAMEGPU->template setVariable<unsigned char>(WAITING_ROOM_FLAG, INSIDE_WAITING_ROOM);
                    FLAMEGPU->template setVariable<unsigned char>(ENTRY_EXIT_FLAG, STAYING_IN_WAITING_ROOM);
                    *stay = 2;

                }
                else if((short) alternative_resources_type_det[agent_type][final_target] == WAITINGROOM && FLAMEGPU->template getVariable<unsigned char>(WAITING_ROOM_FLAG) == INSIDE_WAITING_ROOM){
                    //The agent have waited in waiting room and now go to the right flux room
                    FLAMEGPU->template setVariable<unsigned char>(WAITING_ROOM_FLAG, OUTSIDE_WAITING_ROOM);
                    final_target = FLAMEGPU->template getVariable<short>(NODE_WAITING_FOR);
                    FLAMEGPU->template setVariable<short>(NODE_WAITING_FOR, -1);
                    *available = true;
                }
                else if((short) alternative_resources_type_det[agent_type][final_target] != WAITINGROOM){

                    // Try getting the resources of the room
                    get_specific_resource = ++specific_resources_counter[agent_type][final_target];

                    if(get_specific_resource <= specific_resources[agent_type][final_target]){
                        get_global_resource = ++global_resources_counter[final_target];
                        if(get_global_resource <= global_resources[final_target]){
                            *available = true;
                        } else {
                            get_global_resource = --global_resources_counter[final_target];
                        }
                    }

                    // If the initial room is not avaiable because the resources are over, explore the alternatives:
                    if(!*available){
                        get_specific_resource = --specific_resources_counter[agent_type][final_target];

                        // Search another room of the same type and area
                        if((short) alternative_resources_area_det[agent_type][final_target] == area && (short) alternative_resources_type_det[agent_type][final_target] == flow){
                            final_target = findFreeRoomOfTypeAndArea(FLAMEGPU, flow, random, lenght_rooms, ward_indeces, available);
                        }
                        // Search another room of the alternative
                        else if((short) alternative_resources_area_det[agent_type][final_target] != area || (short) alternative_resources_type_det[agent_type][final_target] != flow){

                            auto messages = FLAMEGPU->message_in((short) alternative_resources_type_det[agent_type][final_target]);

                            unsigned short ward_indeces_alternative[SOLUTION_LENGTH];
                            unsigned short j = 0;
                            unsigned short k = 0;

                            auto i = messages.begin();
                            while(i != messages.end()){
                                const short local_area = (*i).template getVariable<short>(AREA);

                                if(local_area == (short) alternative_resources_area_det[agent_type][final_target]){
                                    ward_indeces_alternative[j] = k;
                                    j++;
                                }

                                i++;
                                k++;
                            }
                            int random = round(cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_TAKE_NEW_DESTINATION_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX)], UNIFORM, contacts_id, 0.0f, (float) (j-1), false));
                            final_target = findFreeRoomOfTypeAndArea(FLAMEGPU, alternative_resources_type_det[agent_type][final_target], random, lenght_rooms, ward_indeces_alternative, available);
                        }
                    }
                }

                if(!*available || (short) alternative_resources_type_det[agent_type][final_target] == -1){
                    *stay = 1;
                    final_target = start_node;
                    if(!CHECK_IS_SPAWNROOM(start_node) && start_node_type != WAITINGROOM) {
                        ++global_resources_counter[start_node];
                        ++specific_resources_counter[agent_type][start_node];
                    }
                }
            }
        }
        else{
            unsigned short spawnroom_area;

            if(identified){
                if(!(FLAMEGPU->template getVariable<unsigned char>(INIT))){
                    const float final_target_coordinates[3] = {FLAMEGPU->template getVariable<float, 3>(FINAL_TARGET, 0), FLAMEGPU->template getVariable<float, 3>(FINAL_TARGET, 1), FLAMEGPU->template getVariable<float, 3>(FINAL_TARGET, 2)};

                    auto coord2index = FLAMEGPU->environment.template getMacroProperty<short, FLOORS, ENV_DIM_Z, ENV_DIM_X>(COORD2INDEX);

                    final_target = (short) coord2index[(unsigned short)(final_target_coordinates[1]/YOFFSET)][(unsigned short)final_target_coordinates[2]][(unsigned short)final_target_coordinates[0]];
                }
                else{
                    int spawnrooms_areas[NUM_AREAS] = {-1};
                    unsigned short num_spawnrooms_areas = 0;

                    for(unsigned int area = 0; area < NUM_AREAS; ++area){
                        if(spawnrooms_areas_ids[area][0] > 0){
                            spawnrooms_areas[num_spawnrooms_areas] = area;
                            num_spawnrooms_areas++;
                        }
                    }

                    spawnroom_area = spawnrooms_areas[(unsigned short) (cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_UNIFORM_0_1_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX)], UNIFORM, contacts_id, 0.0f, (float) (num_spawnrooms_areas-1), false))];
                    final_target = (unsigned short) spawnrooms_areas_ids[spawnroom_area][(unsigned short) (cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_UNIFORM_0_1_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX)], UNIFORM, contacts_id, 1.0f, (float) spawnrooms_areas_ids[spawnroom_area][0], false))];
                }
            }
            else{
                spawnroom_area = flow_area;
                final_target = (unsigned short) spawnrooms_areas_ids[spawnroom_area][(unsigned short) (cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_UNIFORM_0_1_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX)], UNIFORM, contacts_id, 1.0f, (float) spawnrooms_areas_ids[spawnroom_area][0], false))];
            }
        }

        FLAMEGPU->template setVariable<float, 3>(FINAL_TARGET, 0, FLAMEGPU->environment.template getProperty<unsigned short, V>(INDEX2COORDX, final_target));
        FLAMEGPU->template setVariable<float, 3>(FINAL_TARGET, 1, FLAMEGPU->environment.template getProperty<unsigned short, V>(INDEX2COORDY, final_target));
        FLAMEGPU->template setVariable<float, 3>(FINAL_TARGET, 2, FLAMEGPU->environment.template getProperty<unsigned short, V>(INDEX2COORDZ, final_target));

        if(identified == IDENTIFIED)
            FLAMEGPU->template setVariable<int>(ROOM_FOR_QUARANTINE_INDEX, final_target);

        if((short) env_flow[agent_type][agent_subtype][week_day_flow][flow_index + 1] == -1 && CHECK_IS_SPAWNROOM(final_target))
            *stay = 1;

#if defined(DEBUG) && !defined(ENSEMBLE)
        printf("5,%d,%d,Ending of take_new_destination_flow for agent with id %d\n", FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->template getVariable<short>(CONTACTS_ID));
#endif
        return final_target;
    }

    /**
     * Find the shortest path between two nodes in the graph.
    */
    template<typename MessageIn, typename MessageOut>
    FLAMEGPU_DEVICE_FUNCTION void a_star(DeviceAPI<MessageIn, MessageOut>* FLAMEGPU, const unsigned short start, const unsigned short goal, short* solution) {
#if defined(DEBUG) && !defined(ENSEMBLE)
        printf("5,%d,%d,Beginning of a_star for agent with id %d\n", FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX), FLAMEGPU-> getStepCounter(), FLAMEGPU->template getVariable<short>(CONTACTS_ID));
#endif
        short closedset[V];
        short openset[V][3];

        for (unsigned short i = 0; i < V; ++i) {
            closedset[i] = NOT_PRESENT;
            openset[i][0] = NOT_PRESENT;
            openset[i][1] = NOT_PRESENT;
            openset[i][2] = NOT_PRESENT;
        }

        float x_start = FLAMEGPU->environment.template getProperty<unsigned short, V>(INDEX2COORDX, start);
        float x_goal = FLAMEGPU->environment.template getProperty<unsigned short, V>(INDEX2COORDX, goal);
        float z_start = FLAMEGPU->environment.template getProperty<unsigned short, V>(INDEX2COORDZ, start);
        float z_goal = FLAMEGPU->environment.template getProperty<unsigned short, V>(INDEX2COORDZ, goal);

        // Initialize the starting node
        short initial_h = MANHATTAN_DISTANCE(x_start, x_goal, z_start, z_goal);
        openset[start][0] = initial_h;
        openset[start][1] = 0;
        openset[start][2] = STARTING_POINT;

        auto adjmatrix = FLAMEGPU->environment.template getMacroProperty<unsigned short, V, V>(ADJMATRIX);

        // Keep looping WHILE there are elements in the open set
        for(unsigned short n_open = 1; n_open;) {
            //1a. Identify next node to be expanded!
            short next_vertex = NOT_PRESENT;

            for(short i = 0, curr_f_cost; i < V; ++i) {
                if((curr_f_cost = openset[i][F_COST]) != NOT_PRESENT) {
                    if(next_vertex == NOT_PRESENT || curr_f_cost < openset[next_vertex][F_COST]) {
                        next_vertex = i;
                    }
                }
            }

            //1b. Pick the selected node and remove it from the openset
            short curr_node[3];
            curr_node[0] = openset[next_vertex][0];
            curr_node[1] = openset[next_vertex][1];
            curr_node[2] = openset[next_vertex][2];

            openset[next_vertex][0] = NOT_PRESENT;
            openset[next_vertex][1] = NOT_PRESENT;
            openset[next_vertex][2] = NOT_PRESENT;
            --n_open;

            //2. Check if it matches with the goal
            if(next_vertex == goal) {
                short backward_solution[SOLUTION_LENGTH], length = 0;

                closedset[next_vertex] = curr_node[PREV];

                for(short backtrack = closedset[next_vertex]; backtrack != STARTING_POINT; backtrack = closedset[backtrack])
                    backward_solution[length++] = backtrack;

                for(unsigned short i = 0, j = length - 1; i < length; ++i, --j){
                    solution[i] = backward_solution[j];
                }
                solution[length] = goal;
                for(unsigned short i = length+1; i < SOLUTION_LENGTH; ++i)
                    solution[i] = -1;

#if defined(DEBUG) && !defined(ENSEMBLE)
                printf("5,%d,%d,Ending of a_star for agent with id %d\n", FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX), FLAMEGPU-> getStepCounter(), FLAMEGPU->template getVariable<short>(CONTACTS_ID));
#endif
                return;
            }

            //3. Check if it is already visited (is it in the closedset?)
            if(closedset[next_vertex] == NOT_PRESENT) {
                short curr_x = FLAMEGPU->environment.template getProperty<unsigned short, V>(INDEX2COORDX, next_vertex), curr_z = FLAMEGPU->environment.template getProperty<unsigned short, V>(INDEX2COORDZ, next_vertex);
                //4. Add each unvisited neighbor of the current node to the openset
                for(unsigned short i = 0; i < V; ++i) {
                    unsigned short we = adjmatrix[i][next_vertex];
                    // Foreach unvisited neighbor
                    if(we > 0 && closedset[i] == NOT_PRESENT) {
                        short new_g = curr_node[G_COST] + we;
                        short g_cost_neighbor = openset[i][G_COST];
                        /* A new node is added to the open set if
                        it not present yet in the openset,
                        or if the new cost is less than the current present in the openset.  */
                        bool first_time = g_cost_neighbor == NOT_PRESENT,
                            not_new_but_interesting = new_g < g_cost_neighbor;

                        //If it is the first time, the node is added to the openset and the number of elements is ++increased
                        if((first_time && ++n_open) || not_new_but_interesting) {
                            // Add new node or replace node in openset estimating f(n)
                            float x_coord_i = FLAMEGPU->environment.template getProperty<unsigned short, V>(INDEX2COORDX, i);
                            float z_coord_i = FLAMEGPU->environment.template getProperty<unsigned short, V>(INDEX2COORDZ, i);
                            openset[i][0] = new_g + MANHATTAN_DISTANCE(x_coord_i, x_goal, z_coord_i, z_goal);
                            openset[i][1] = new_g;
                            openset[i][2] = next_vertex;
                        }
                    }
                }

                closedset[next_vertex] = curr_node[PREV];
            }
        }

#if defined(DEBUG) && !defined(ENSEMBLE)
        printf("5,%d,%d,Ending of a_star for agent with id %d\n", FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX), FLAMEGPU-> getStepCounter(), FLAMEGPU->template getVariable<short>(CONTACTS_ID));
#endif
    }

    /**
     * Update agent intermediate and final targets.
    */
    template<typename MessageIn, typename MessageOut>
    FLAMEGPU_DEVICE_FUNCTION void update_targets(DeviceAPI<MessageIn, MessageOut>* FLAMEGPU, short* new_targets, unsigned short *target_index, const bool clean, const int stay) {
#if defined(DEBUG) && !defined(ENSEMBLE)
        printf("5,%d,%d,Beginning of update_targets for agent with id %d\n", FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->template getVariable<short>(CONTACTS_ID));
#endif
        auto intermediate_target_x = FLAMEGPU->environment.template getMacroProperty<float, TOTAL_AGENTS_ESTIMATION, SOLUTION_LENGTH>(INTERMEDIATE_TARGET_X);
        auto intermediate_target_y = FLAMEGPU->environment.template getMacroProperty<float, TOTAL_AGENTS_ESTIMATION, SOLUTION_LENGTH>(INTERMEDIATE_TARGET_Y);
        auto intermediate_target_z = FLAMEGPU->environment.template getMacroProperty<float, TOTAL_AGENTS_ESTIMATION, SOLUTION_LENGTH>(INTERMEDIATE_TARGET_Z);
        auto stay_matrix = FLAMEGPU->environment.template getMacroProperty<unsigned int, TOTAL_AGENTS_ESTIMATION, SOLUTION_LENGTH>(STAY);

        const short contacts_id = FLAMEGPU->template getVariable<short>(CONTACTS_ID);

        float new_target_x, new_target_y, new_target_z;

        if(clean){
            const unsigned short next_index = FLAMEGPU->template getVariable<unsigned short>(NEXT_INDEX);

            if(next_index != *target_index){
                *target_index = (next_index + 1) % SOLUTION_LENGTH;
            }

            stay_matrix[contacts_id][*target_index].exchange(0);
        }

        short i = 1;
        while(i < SOLUTION_LENGTH && new_targets[i] != -1){
            float x = FLAMEGPU->environment.template getProperty<float, V>(NODE_X, new_targets[i]);
            float z = FLAMEGPU->environment.template getProperty<float, V>(NODE_Z, new_targets[i]);

            new_target_x = x;
            new_target_y = FLAMEGPU->environment.template getProperty<unsigned short, V>(INDEX2COORDY, new_targets[i]);
            new_target_z = z;

            intermediate_target_x[contacts_id][*target_index].exchange(new_target_x);
            intermediate_target_y[contacts_id][*target_index].exchange(new_target_y);
            intermediate_target_z[contacts_id][*target_index].exchange(new_target_z);

            if(i > 1)
                stay_matrix[contacts_id][*target_index].exchange(0);

            *target_index = (*target_index + 1) % SOLUTION_LENGTH;

            ++i;
        }

        stay_matrix[contacts_id][*target_index].exchange(stay);
        FLAMEGPU->template setVariable<unsigned short>(TARGET_INDEX, *target_index);

#if defined(DEBUG) && !defined(ENSEMBLE)
        printf("5,%d,%d,Ending update_targets for agent with id %d\n", FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->template getVariable<short>(CONTACTS_ID));
#endif
    }

    /**
     * Update agent's flow for the next day in which the agent will enter in the environment.
    */
    template<typename MessageIn, typename MessageOut>
    FLAMEGPU_DEVICE_FUNCTION void update_flow(DeviceAPI<MessageIn, MessageOut> *FLAMEGPU, const bool quarantine){
#if defined(DEBUG) && !defined(ENSEMBLE)
        printf("5,%d,%d,Beginning of update_flow for agent with id %d\n", FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->template getVariable<short>(CONTACTS_ID));
#endif
        const short agent_type = FLAMEGPU->template getVariable<short>(AGENT_TYPE);
        const short agent_subtype = FLAMEGPU->template getVariable<short>(AGENT_SUBTYPE);
        const short contacts_id = FLAMEGPU->template getVariable<short>(CONTACTS_ID);
        const unsigned short target_index = FLAMEGPU->template getVariable<unsigned short>(TARGET_INDEX);
        const unsigned char identified = FLAMEGPU->template getVariable<unsigned char>(IDENTIFIED_INFECTED);

        auto stay_matrix = FLAMEGPU->environment.template getMacroProperty<unsigned int, TOTAL_AGENTS_ESTIMATION, SOLUTION_LENGTH>(STAY);
        auto env_flow = FLAMEGPU->environment.template getMacroProperty<short, NUMBER_OF_AGENTS_TYPES, NUMBER_OF_AGENTS_SUBTYPES, DAYS_IN_A_WEEK, FLOW_LENGTH>(ENV_FLOW);
        auto env_flow_distr = FLAMEGPU->environment.template getMacroProperty<short, NUMBER_OF_AGENTS_TYPES, NUMBER_OF_AGENTS_SUBTYPES, DAYS_IN_A_WEEK, FLOW_LENGTH>(ENV_FLOW_DISTR);
        auto env_flow_distr_firstparam = FLAMEGPU->environment.template getMacroProperty<int, NUMBER_OF_AGENTS_TYPES, NUMBER_OF_AGENTS_SUBTYPES, DAYS_IN_A_WEEK, FLOW_LENGTH>(ENV_FLOW_DISTR_FIRSTPARAM);
        auto env_flow_distr_secondparam = FLAMEGPU->environment.template getMacroProperty<int, NUMBER_OF_AGENTS_TYPES, NUMBER_OF_AGENTS_SUBTYPES, DAYS_IN_A_WEEK, FLOW_LENGTH>(ENV_FLOW_DISTR_SECONDPARAM);
        auto env_hours_schedule = FLAMEGPU->environment.template getMacroProperty<int, NUMBER_OF_AGENTS_TYPES, NUMBER_OF_AGENTS_SUBTYPES, DAYS_IN_A_WEEK, HOURS_SCHEDULE_LENGTH>(ENV_HOURS_SCHEDULE);

        unsigned char entry_time_index = FLAMEGPU->template getVariable<unsigned char>(ENTRY_TIME_INDEX) + 1;
        unsigned char week_day_flow = FLAMEGPU->template getVariable<unsigned char>(WEEK_DAY_FLOW);
        unsigned char empty_days;
        int start_step;

        if((int) env_hours_schedule[agent_type][agent_subtype][week_day_flow][2 * entry_time_index] == 0 || quarantine){
            entry_time_index = 0;
            week_day_flow = (week_day_flow + 1) % DAYS_IN_A_WEEK;

            empty_days = 0;
            while((short) env_flow[agent_type][agent_subtype][week_day_flow][0] == -1){
                empty_days++;
                week_day_flow = (week_day_flow + 1) % DAYS_IN_A_WEEK;
            }

            start_step = (int) env_hours_schedule[agent_type][agent_subtype][week_day_flow][0];
        }
        else{
            start_step = ((int) env_hours_schedule[agent_type][agent_subtype][week_day_flow][2 * entry_time_index] - ((FLAMEGPU->getStepCounter() + START_STEP_TIME) % STEPS_IN_A_DAY));
            start_step = start_step < 1 ? 1: start_step;
        }

        if(entry_time_index == 0){
            unsigned int stay_flow_index_0 = (unsigned int) cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_FLOW_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX)], (short) env_flow_distr[agent_type][agent_subtype][week_day_flow][0], contacts_id, (float) env_flow_distr_firstparam[agent_type][agent_subtype][week_day_flow][0], (float) env_flow_distr_secondparam[agent_type][agent_subtype][week_day_flow][0], true);
            stay_matrix[contacts_id][target_index].exchange((unsigned int) (empty_days * STEPS_IN_A_DAY + (STEPS_IN_A_DAY - ((FLAMEGPU->getStepCounter() + START_STEP_TIME) % STEPS_IN_A_DAY)) + start_step + stay_flow_index_0));
            FLAMEGPU->template setVariable<unsigned short>(FLOW_INDEX, 0);
        }
        else{
            stay_matrix[contacts_id][target_index].exchange((unsigned int) start_step);
        }

        FLAMEGPU->template setVariable<unsigned char>(ENTRY_TIME_INDEX, entry_time_index);
        FLAMEGPU->template setVariable<unsigned char>(WEEK_DAY_FLOW, week_day_flow);
#if defined(DEBUG) && !defined(ENSEMBLE)
        printf("5,%d,%d,Ending update_flow for agent with id %d\n", FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->template getVariable<short>(CONTACTS_ID));
#endif
    }

    template<typename MessageOut>
    FLAMEGPU_DEVICE_FUNCTION void put_in_quarantine(DeviceAPI<MessageBucket, MessageOut> *FLAMEGPU){
#if defined(DEBUG) && !defined(ENSEMBLE)
        printf("5,%d,%d,Beginning put_in_quarantine for agent with id %d\n", FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->template getVariable<short>(CONTACTS_ID));
#endif
        const unsigned short day = FLAMEGPU->environment.template getProperty<unsigned short>(DAY);
        const short contacts_id = FLAMEGPU->template getVariable<short>(CONTACTS_ID);
        const short agent_type = FLAMEGPU->template getVariable<short>(AGENT_TYPE);

        auto stay_matrix = FLAMEGPU->environment.template getMacroProperty<unsigned int, TOTAL_AGENTS_ESTIMATION, SOLUTION_LENGTH>(STAY);
        auto env_quarantine_days_distr = FLAMEGPU->environment.template getMacroProperty<short, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_QUARANTINE_DAYS_DISTR);
        auto env_quarantine_days_distr_firstparam = FLAMEGPU->environment.template getMacroProperty<int, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_QUARANTINE_DAYS_DISTR_FIRSTPARAM);
        auto env_quarantine_days_distr_secondparam = FLAMEGPU->environment.template getMacroProperty<int, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_QUARANTINE_DAYS_DISTR_SECONDPARAM);
        auto env_quarantine_swab_days_distr = FLAMEGPU->environment.template getMacroProperty<short, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_QUARANTINE_SWAB_DAYS_DISTR);
        auto env_quarantine_swab_days_distr_firstparam = FLAMEGPU->environment.template getMacroProperty<float, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_QUARANTINE_SWAB_DAYS_DISTR_FIRSTPARAM);
        auto env_quarantine_swab_days_distr_secondparam = FLAMEGPU->environment.template getMacroProperty<float, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_QUARANTINE_SWAB_DAYS_DISTR_SECONDPARAM);

        unsigned char quarantine = FLAMEGPU->template getVariable<unsigned char>(QUARANTINE);
        unsigned char identified_bool = FLAMEGPU->template getVariable<unsigned char>(IDENTIFIED_INFECTED);
        unsigned char severity = FLAMEGPU->template getVariable<unsigned char>(SEVERITY);
        unsigned short target_index = FLAMEGPU->template getVariable<unsigned short>(TARGET_INDEX);
        bool already_in_quarantine = quarantine > 0;

        quarantine = cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_QUARANTINE_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX)], (short) env_quarantine_days_distr[day-1][agent_type], contacts_id, (float) env_quarantine_days_distr_firstparam[day-1][agent_type], (float) env_quarantine_days_distr_secondparam[day-1][agent_type], true);
        FLAMEGPU->template setVariable<unsigned char>(QUARANTINE, quarantine);

        int stay = 1;

        auto coord2index = FLAMEGPU->environment.template getMacroProperty<short, FLOORS, ENV_DIM_Z, ENV_DIM_X>(COORD2INDEX);

        const float final_target[3] = {FLAMEGPU->template getVariable<float, 3>(FINAL_TARGET, 0), FLAMEGPU->template getVariable<float, 3>(FINAL_TARGET, 1), FLAMEGPU->template getVariable<float, 3>(FINAL_TARGET, 2)};
        const unsigned char agent_with_a_rate = FLAMEGPU->template getVariable<unsigned char>(AGENT_WITH_A_RATE);

        stay_matrix[contacts_id][target_index].exchange(0);

        auto global_resources_counter = FLAMEGPU->environment.template getMacroProperty<unsigned int, V>(GLOBAL_RESOURCES_COUNTER);
        auto specific_resources_counter = FLAMEGPU->environment.template getMacroProperty<unsigned int, NUMBER_OF_AGENTS_TYPES, V>(SPECIFIC_RESOURCES_COUNTER);

        if(!already_in_quarantine){
            const short start_node = coord2index[(unsigned short)(final_target[1]/YOFFSET)][(unsigned short)final_target[2]][(unsigned short)final_target[0]];
            const short start_node_type = FLAMEGPU->environment.template getProperty<short, V>(NODE_TYPE, start_node);

            if(!CHECK_IS_SPAWNROOM(start_node) && start_node_type != WAITINGROOM){
                --global_resources_counter[start_node];
                --specific_resources_counter[agent_type][start_node];
            }

            bool available = false;
            const short quarantine_node = take_new_destination_flow(FLAMEGPU, &stay, start_node, &available, identified_bool, severity);

            auto counters = FLAMEGPU->environment.template getMacroProperty<unsigned int, NUM_COUNTERS>(COUNTERS);
            auto spawnrooms_areas_ids = FLAMEGPU->environment.template getMacroProperty<unsigned short, NUM_AREAS, NUM_SPAWNROOM + 1>(SPAWNROOMS_AREAS_IDS);

            if(!CHECK_IS_SPAWNROOM(quarantine_node) && !FLAMEGPU->template getVariable<unsigned char>(INIT)){
                FLAMEGPU->template setVariable<unsigned char>(INIT, 1);

                int spawnrooms_areas[NUM_AREAS] = {-1};
                unsigned short num_spawnrooms_areas = 0;

                for(unsigned int area = 0; area < NUM_AREAS; ++area){
                    if(spawnrooms_areas_ids[area][0] > 0){
                        spawnrooms_areas[num_spawnrooms_areas] = area;
                        num_spawnrooms_areas++;
                    }
                }

                unsigned short random_area = spawnrooms_areas[(unsigned short) (cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_UNIFORM_0_1_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX)], UNIFORM, contacts_id, 0.0f, (float) (num_spawnrooms_areas-1), false))];
                unsigned short extern_node =  (unsigned short) spawnrooms_areas_ids[random_area][(unsigned short) (cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_UNIFORM_0_1_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX)], UNIFORM, contacts_id, 1.0f, (float) spawnrooms_areas_ids[random_area][0], false))];

                float x = cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_OFFSET_X_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX)], UNIFORM, contacts_id, FLAMEGPU->environment.template getProperty<float, NUM_SPAWNROOM * 4>(EXTERN_RANGES, extern_node * 2), FLAMEGPU->environment.template getProperty<float, NUM_SPAWNROOM * 4>(EXTERN_RANGES, (extern_node * 2) + 1), false);
                float y = FLAMEGPU->environment.template getProperty<unsigned short, NUM_SPAWNROOM + 1>(ENTRANCE_Y_COORDS, extern_node);
                float z = cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_OFFSET_Z_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX)], UNIFORM, contacts_id, FLAMEGPU->environment.template getProperty<float, NUM_SPAWNROOM * 4>(EXTERN_RANGES, (extern_node * 2) + 2 * NUM_SPAWNROOM), FLAMEGPU->environment.template getProperty<float, NUM_SPAWNROOM * 4>(EXTERN_RANGES, (extern_node * 2) + 2 * NUM_SPAWNROOM + 1), false);

                FLAMEGPU->template setVariable<float>(X, x);
                FLAMEGPU->template setVariable<float>(Y, y);
                FLAMEGPU->template setVariable<float>(Z, z);

                FLAMEGPU->template setVariable<float, 3>(FINAL_TARGET, 0, x);
                FLAMEGPU->template setVariable<float, 3>(FINAL_TARGET, 1, y);
                FLAMEGPU->template setVariable<float, 3>(FINAL_TARGET, 2, z);
            }

            short solution_start_quarantine[SOLUTION_LENGTH] = {-1};

            a_star(FLAMEGPU, start_node, quarantine_node, solution_start_quarantine);

            update_targets(FLAMEGPU, solution_start_quarantine, &target_index, false, quarantine * STEPS_IN_A_DAY);

            counters[AGENTS_IN_QUARANTINE]++;
        }
        else{
            stay_matrix[contacts_id][target_index].exchange(quarantine * STEPS_IN_A_DAY);
        }

        if(agent_with_a_rate)
            FLAMEGPU->template setVariable<unsigned short>(FLOW_INDEX, 0);

        if((short) env_quarantine_swab_days_distr[day-1][agent_type] != NO_SWAB){
            int swab_steps = round(cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_QUARANTINE_SWAB_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX)], (short) env_quarantine_swab_days_distr[day-1][agent_type], contacts_id, (float) (STEPS_IN_A_DAY * env_quarantine_swab_days_distr_firstparam[day-1][agent_type]), (float) (STEPS_IN_A_DAY * env_quarantine_swab_days_distr_secondparam[day-1][agent_type]), true));
            FLAMEGPU->template setVariable<int>(SWAB_STEPS, swab_steps);
        }

#if defined(DEBUG) && !defined(ENSEMBLE)
        printf("5,%d,%d,Ending put_in_quarantine for agent with id %d\n", FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->template getVariable<short>(CONTACTS_ID));
#endif
    }

    /**
     * Make a swab to the agent and handle quarantine.
    */
    template<typename MessageOut>
    FLAMEGPU_DEVICE_FUNCTION void swab(DeviceAPI<MessageBucket, MessageOut> *FLAMEGPU){
#if defined(DEBUG) && !defined(ENSEMBLE)
        printf("5,%d,%d,Beginning swab for agent with id %d\n", FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->template getVariable<short>(CONTACTS_ID));
#endif
        const unsigned short day = FLAMEGPU->environment.template getProperty<unsigned short>(DAY);
        const unsigned char disease_state = FLAMEGPU->template getVariable<unsigned char>(DISEASE_STATE);
        const short contacts_id = FLAMEGPU->template getVariable<short>(CONTACTS_ID);
        const short agent_type = FLAMEGPU->template getVariable<short>(AGENT_TYPE);

        auto stay_matrix = FLAMEGPU->environment.template getMacroProperty<unsigned int, TOTAL_AGENTS_ESTIMATION, SOLUTION_LENGTH>(STAY);
        auto env_swab_sensitivity = FLAMEGPU->environment.template getMacroProperty<float, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_SWAB_SENSITIVITY);
        auto env_swab_specificity = FLAMEGPU->environment.template getMacroProperty<float, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_SWAB_SPECIFICITY);
        auto env_swab_distr = FLAMEGPU->environment.template getMacroProperty<short, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_SWAB_DISTR);
        auto env_swab_distr_firstparam = FLAMEGPU->environment.template getMacroProperty<float, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_SWAB_DISTR_FIRSTPARAM);
        auto env_swab_distr_secondparam = FLAMEGPU->environment.template getMacroProperty<float, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_SWAB_DISTR_SECONDPARAM);
        auto env_quarantine_days_distr = FLAMEGPU->environment.template getMacroProperty<short, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_QUARANTINE_DAYS_DISTR);
        auto env_quarantine_swab_sensitivity = FLAMEGPU->environment.template getMacroProperty<float, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_QUARANTINE_SWAB_SENSITIVITY);
        auto env_quarantine_swab_specificity = FLAMEGPU->environment.template getMacroProperty<float, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_QUARANTINE_SWAB_SPECIFICITY);
        auto counters = FLAMEGPU->environment.template getMacroProperty<unsigned int, NUM_COUNTERS>(COUNTERS);

        unsigned char identified_bool = FLAMEGPU->template getVariable<unsigned char>(IDENTIFIED_INFECTED);
        unsigned char quarantine = FLAMEGPU->template getVariable<unsigned char>(QUARANTINE);
        unsigned char severity = FLAMEGPU->template getVariable<unsigned char>(SEVERITY);
        unsigned short target_index = FLAMEGPU->template getVariable<unsigned short>(TARGET_INDEX);
        bool already_in_quarantine = quarantine > 0;

        if(disease_state == INFECTED){
            const float sensitivity_swab = already_in_quarantine ? (float) env_quarantine_swab_sensitivity[day-1][agent_type]: (float) env_swab_sensitivity[day-1][agent_type];
            float random_sensitivity = cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_UNIFORM_0_1_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX)], UNIFORM, contacts_id, 0.0f, 1.0f, false);

            if(random_sensitivity < sensitivity_swab){
                // True positive
                if(!already_in_quarantine){
                    const unsigned char risk_class = FLAMEGPU->template getVariable<unsigned char>(RISK_CLASS);
                    const float severity_covid = FLAMEGPU->environment.template getProperty<float, RISK_CLASSES + 1>(VIRUS_SEVERITY, risk_class);

                    float random_severity = cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_UNIFORM_0_1_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX)], UNIFORM, contacts_id, 0.0f, 1.0f, false);

                    if(random_severity < severity_covid)
                        severity = MAJOR;
                }
                identified_bool = IDENTIFIED;
            }
            else{
                // False negative if random_sensitivity is greater (>) than sensitivity_swab
                identified_bool = NOT_IDENTIFIED;
            }
        }
        else {
            // False positive
            const float specificity_swab = already_in_quarantine ? (float) env_quarantine_swab_specificity[day-1][agent_type]: (float) env_swab_specificity[day-1][agent_type];
            float random_specificity = cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_UNIFORM_0_1_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX)], UNIFORM, contacts_id, 0.0f, 1.0f, false);

            if(random_specificity >= specificity_swab){
                identified_bool = IDENTIFIED;
            }
            else{
                // True negative if random_specificity less or equal (<=) than specificity_swab
                identified_bool = NOT_IDENTIFIED;
            }
        }

        FLAMEGPU->template setVariable<unsigned char>(IDENTIFIED_INFECTED, identified_bool);
        FLAMEGPU->template setVariable<unsigned char>(SEVERITY, severity);

        counters[SWABS]++;

        FLAMEGPU->template setVariable<int>(SWAB_STEPS, 0);

        // Now the agent could has been identified as infected
        if(identified_bool == IDENTIFIED){
            /**
             * Identified (True or False Positive), we have different cases:
             *  - No quarantine: do nothing
             *  - Quarantine and no swab during quarantine: put the agent in quarantine in the given room for n days
             *                                              (where n is generated using the selected distribution and parameters).
             *  - Quarantine and swab during quarantine: put the agent in quarantine in the given room for n days
             *                                           (where n is generated using the selected distribution and parameters) and
             *                                           make a swab every m days (where m is generated using the selected distribution
             *                                           and parameters). A Negative swab will allow the agent to exit from the quarantine.
             */
            if((short) env_quarantine_days_distr[day-1][agent_type] != NO_QUARANTINE){
                put_in_quarantine(FLAMEGPU);
            }
        }
        else{
            // Not identified (True or False Negative): the agent exits from quarantine
            if(already_in_quarantine){
                stay_matrix[contacts_id][target_index].exchange(1);
            }
            else{
                if((short) env_swab_distr[day-1][agent_type] != NO_SWAB){
                    int swab_steps = round(cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_SWAB_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX)], (short) env_swab_distr[day-1][agent_type], contacts_id, (float) (STEPS_IN_A_DAY * env_swab_distr_firstparam[day-1][agent_type]), (float) (STEPS_IN_A_DAY * env_swab_distr_secondparam[day-1][agent_type]), true));
                    FLAMEGPU->template setVariable<int>(SWAB_STEPS, swab_steps);
                }
            }
        }

#if defined(DEBUG) && !defined(ENSEMBLE)
        printf("5,%d,%d,Ending swab for agent with id %d\n", FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->template getVariable<short>(CONTACTS_ID));
#endif
    }

    /**
     * The agent exits from quarantine.
    */
    template<typename MessageIn, typename MessageOut>
    FLAMEGPU_DEVICE_FUNCTION void exit_from_quarantine(DeviceAPI<MessageIn, MessageOut> *FLAMEGPU){
#if defined(DEBUG) && !defined(ENSEMBLE)
        printf("5,%d,%d,Beginning exit_from_quarantine for agent with id %d\n", FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->template getVariable<short>(CONTACTS_ID));
#endif
        const unsigned short day = FLAMEGPU->environment.template getProperty<unsigned short>(DAY);
        const unsigned char agent_with_a_rate = FLAMEGPU->template getVariable<unsigned char>(AGENT_WITH_A_RATE);
        const short contacts_id = FLAMEGPU->template getVariable<short>(CONTACTS_ID);
        const short agent_type = FLAMEGPU->template getVariable<short>(AGENT_TYPE);
        const short agent_subtype = FLAMEGPU->template getVariable<short>(AGENT_SUBTYPE);
        const int quarantine_node = FLAMEGPU->template getVariable<int>(ROOM_FOR_QUARANTINE_INDEX);

        short solution_quarantine_extern[SOLUTION_LENGTH] = {-1};
        unsigned short target_index = FLAMEGPU->template getVariable<unsigned short>(TARGET_INDEX);

        auto spawnrooms_areas_ids = FLAMEGPU->environment.template getMacroProperty<unsigned short, NUM_AREAS, NUM_SPAWNROOM + 1>(SPAWNROOMS_AREAS_IDS);
        auto stay_matrix = FLAMEGPU->environment.template getMacroProperty<unsigned int, TOTAL_AGENTS_ESTIMATION, SOLUTION_LENGTH>(STAY);
        auto env_hours_schedule = FLAMEGPU->environment.template getMacroProperty<int, NUMBER_OF_AGENTS_TYPES, NUMBER_OF_AGENTS_SUBTYPES, DAYS_IN_A_WEEK, HOURS_SCHEDULE_LENGTH>(ENV_HOURS_SCHEDULE);
        auto env_swab_distr = FLAMEGPU->environment.template getMacroProperty<short, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_SWAB_DISTR);
        auto env_swab_distr_firstparam = FLAMEGPU->environment.template getMacroProperty<float, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_SWAB_DISTR_FIRSTPARAM);
        auto env_swab_distr_secondparam = FLAMEGPU->environment.template getMacroProperty<float, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_SWAB_DISTR_SECONDPARAM);
        auto counters = FLAMEGPU->environment.template getMacroProperty<unsigned int, NUM_COUNTERS>(COUNTERS);
        auto coord2index = FLAMEGPU->environment.template getMacroProperty<short, FLOORS, ENV_DIM_Z, ENV_DIM_X>(COORD2INDEX);

        FLAMEGPU->template setVariable<unsigned char>(SEVERITY, MINOR);
        FLAMEGPU->template setVariable<int>(SWAB_STEPS, 0);
        FLAMEGPU->template setVariable<int>(ROOM_FOR_QUARANTINE_INDEX, -1);
        FLAMEGPU->template setVariable<unsigned char>(QUARANTINE, 0);
        FLAMEGPU->template setVariable<unsigned char>(IDENTIFIED_INFECTED, NOT_IDENTIFIED);

        int spawnrooms_areas[NUM_AREAS] = {-1};
        unsigned short num_spawnrooms_areas = 0;

        for(unsigned int area = 0; area < NUM_AREAS; ++area){
            if(spawnrooms_areas_ids[area][0] > 0){
                spawnrooms_areas[num_spawnrooms_areas] = area;
                num_spawnrooms_areas++;
            }
        }

        unsigned short random_area = spawnrooms_areas[(unsigned short) (cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_UNIFORM_0_1_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX)], UNIFORM, contacts_id, 0.0f, (float) (num_spawnrooms_areas-1), false))];
        unsigned short extern_node =  (unsigned short) spawnrooms_areas_ids[random_area][(unsigned short) (cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_UNIFORM_0_1_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX)], UNIFORM, contacts_id, 1.0f, (float) spawnrooms_areas_ids[random_area][0], false))];

        a_star(FLAMEGPU, quarantine_node, extern_node, solution_quarantine_extern);

        if(!agent_with_a_rate){
            if(!CHECK_IS_SPAWNROOM(quarantine_node))
                update_targets(FLAMEGPU, solution_quarantine_extern, &target_index, false, 1);

            update_flow(FLAMEGPU, true);
        }
        else{
            unsigned short week_day = (FLAMEGPU->environment.template getProperty<unsigned short>(WEEK_DAY) + 1) % DAYS_IN_A_WEEK;

            unsigned short empty_days = 0;
            unsigned short agent_index = week_day;
            while((int) env_hours_schedule[agent_type][agent_subtype][agent_index][1] - (int) env_hours_schedule[agent_type][agent_subtype][agent_index][0] == 0){
                empty_days++;
                agent_index = (agent_index + 1) % DAYS_IN_A_WEEK;
            }

            const unsigned short start_step = (unsigned short) cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_HOURS_SCHEDULE_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX)], UNIFORM, contacts_id, (float) env_hours_schedule[agent_type][agent_subtype][agent_index][0], (float) env_hours_schedule[agent_type][agent_subtype][agent_index][1], true);

            int stay_spawnroom = (STEPS_IN_A_DAY - ((FLAMEGPU->getStepCounter() + START_STEP_TIME) % STEPS_IN_A_DAY)) + start_step;

            if(!CHECK_IS_SPAWNROOM(quarantine_node))
                update_targets(FLAMEGPU, solution_quarantine_extern, &target_index, false, stay_spawnroom);
            else
                stay_matrix[contacts_id][target_index].exchange(stay_spawnroom);
        }

        if((short) env_swab_distr[day-1][agent_type] != NO_SWAB){
            int swab_steps = round(cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_SWAB_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX)], (short) env_swab_distr[day-1][agent_type], contacts_id, (float) (STEPS_IN_A_DAY * env_swab_distr_firstparam[day-1][agent_type]), (float) (STEPS_IN_A_DAY * env_swab_distr_secondparam[day-1][agent_type]), true));
            FLAMEGPU->template setVariable<int>(SWAB_STEPS, swab_steps);
        }

        FLAMEGPU->template setVariable<float, 3>(FINAL_TARGET, 0, FLAMEGPU->environment.template getProperty<unsigned short, V>(INDEX2COORDX, extern_node));
        FLAMEGPU->template setVariable<float, 3>(FINAL_TARGET, 1, FLAMEGPU->environment.template getProperty<unsigned short, V>(INDEX2COORDY, extern_node));
        FLAMEGPU->template setVariable<float, 3>(FINAL_TARGET, 2, FLAMEGPU->environment.template getProperty<unsigned short, V>(INDEX2COORDZ, extern_node));

        float agent_pos[3] = {FLAMEGPU->template getVariable<float>(X), FLAMEGPU->template getVariable<float>(Y), FLAMEGPU->template getVariable<float>(Z)};

        if(agent_pos[1] == INVISIBLE_AGENT_Y)
            printf("0,%d,%d,%d,%d,%f,%f,%f,%d,-1\n", FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->template getVariable<short>(CONTACTS_ID), FLAMEGPU->template getVariable<short>(AGENT_TYPE), agent_pos[0], INVISIBLE_AGENT_Y, agent_pos[2], FLAMEGPU->template getVariable<unsigned char>(DISEASE_STATE));
        else
            printf("0,%d,%d,%d,%d,%f,%f,%f,%d,%d\n", FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->template getVariable<short>(CONTACTS_ID), FLAMEGPU->template getVariable<short>(AGENT_TYPE), agent_pos[0], agent_pos[1], agent_pos[2], FLAMEGPU->template getVariable<unsigned char>(DISEASE_STATE), (short) coord2index[(unsigned short)(agent_pos[1]/YOFFSET)][(unsigned short)agent_pos[2]][(unsigned short)agent_pos[0]]);

        if(!CHECK_IS_SPAWNROOM(quarantine_node))
            FLAMEGPU->template setVariable<unsigned char>(JUST_EXITED_FROM_QUARANTINE, 1);

        counters[AGENTS_IN_QUARANTINE]--;
#if defined(DEBUG) && !defined(ENSEMBLE)
        printf("5,%d,%d,Ending exit_from_quarantine for agent with id %d\n", FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->template getVariable<short>(CONTACTS_ID));
#endif
    }

    /**
     * Handle internal screening.
    */
    template<typename MessageOut>
    FLAMEGPU_DEVICE_FUNCTION void screening(DeviceAPI<MessageBucket, MessageOut> *FLAMEGPU){
#if defined(DEBUG) && !defined(ENSEMBLE)
        printf("5,%d,%d,Beginning screening for agent with id %d\n", FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->template getVariable<short>(CONTACTS_ID));
#endif
        const unsigned short day = FLAMEGPU->environment.template getProperty<unsigned short>(DAY);
        const unsigned char identified_bool = FLAMEGPU->template getVariable<unsigned char>(IDENTIFIED_INFECTED);
        const short agent_type = FLAMEGPU->template getVariable<short>(AGENT_TYPE);

        auto env_swab_distr = FLAMEGPU->environment.template getMacroProperty<short, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_SWAB_DISTR);
        auto env_quarantine_swab_days_distr = FLAMEGPU->environment.template getMacroProperty<short, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_QUARANTINE_SWAB_DAYS_DISTR);

        int swab_steps = FLAMEGPU->template getVariable<int>(SWAB_STEPS);

        swab_steps = swab_steps - 1;
        if(((short) env_swab_distr[day-1][agent_type] != NO_SWAB || (short) env_quarantine_swab_days_distr[day-1][agent_type] != NO_QUARANTINE_SWAB) && swab_steps != -1){
            if(swab_steps){
                FLAMEGPU->template setVariable<int>(SWAB_STEPS, swab_steps);
            }
            else{
                swab(FLAMEGPU);
            }
        }
#if defined(DEBUG) && !defined(ENSEMBLE)
        printf("5,%d,%d,Ending screening for agent with id %d\n", FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->template getVariable<short>(CONTACTS_ID));
#endif
    }

    /**
     * Handle external screening.
    */
    template<typename MessageOut>
    FLAMEGPU_DEVICE_FUNCTION void external_screening(DeviceAPI<MessageBucket, MessageOut> *FLAMEGPU){
#if defined(DEBUG) && !defined(ENSEMBLE)
        printf("5,%d,%d,Beginning external_screening for agent with id %d\n", FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->template getVariable<short>(CONTACTS_ID));
#endif
        const unsigned short day = FLAMEGPU->environment.template getProperty<unsigned short>(DAY);
        const unsigned char disease_state = FLAMEGPU->template getVariable<unsigned char>(DISEASE_STATE);
        const short contacts_id = FLAMEGPU->template getVariable<short>(CONTACTS_ID);
        const short agent_type = FLAMEGPU->template getVariable<short>(AGENT_TYPE);

        auto env_external_screening_first = FLAMEGPU->environment.template getMacroProperty<float, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_EXTERNAL_SCREENING_FIRST);
        auto env_external_screening_second = FLAMEGPU->environment.template getMacroProperty<float, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_EXTERNAL_SCREENING_SECOND);

        unsigned char identified_bool = FLAMEGPU->template getVariable<unsigned char>(IDENTIFIED_INFECTED);

        if(identified_bool != IDENTIFIED && FLAMEGPU->template getVariable<unsigned char>(EXITED_FROM_ENVIRONMENT)){
            float random_external_screening_first = cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_UNIFORM_0_1_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX)], UNIFORM, contacts_id, 0.0f, 1.0f, false);
            if(random_external_screening_first < (float) env_external_screening_first[day-1][agent_type]){
                swab(FLAMEGPU);
#if defined(DEBUG) && !defined(ENSEMBLE)
                printf("5,%d,%d,Ending external_screening for agent with id %d\n", FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->template getVariable<short>(CONTACTS_ID));
#endif
                return;
            }

            float random_external_screening_second = cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_UNIFORM_0_1_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX)], UNIFORM, contacts_id, 0.0f, 1.0f, false);
            if(disease_state == INFECTED && random_external_screening_second < (float) env_external_screening_second[day-1][agent_type]){
                swab(FLAMEGPU);
            }
        }
#if defined(DEBUG) && !defined(ENSEMBLE)
        printf("5,%d,%d,Ending external_screening for agent with id %d\n", FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->template getVariable<short>(CONTACTS_ID));
#endif
    }

    /**
     * Handle contagion processes.
    */
    template<typename MessageIn, typename MessageOut>
    FLAMEGPU_DEVICE_FUNCTION flamegpu::AGENT_STATUS contagion_processes(DeviceAPI<MessageIn, MessageOut> *FLAMEGPU){
#if defined(DEBUG) && !defined(ENSEMBLE)
        printf("5,%d,%d,Beginning of contagion_processes for agent with id %d\n", FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->template getVariable<short>(CONTACTS_ID));
#endif
        const short contacts_id = FLAMEGPU->template getVariable<short>(CONTACTS_ID);
        const unsigned char risk_class = FLAMEGPU->template getVariable<unsigned char>(RISK_CLASS);

        int disease_state = FLAMEGPU->template getVariable<unsigned char>(DISEASE_STATE);
        unsigned short incubation_days = FLAMEGPU->template getVariable<unsigned short>(INCUBATION_DAYS);
        unsigned short infection_days = FLAMEGPU->template getVariable<unsigned short>(INFECTION_DAYS);
        unsigned short fatality_days = FLAMEGPU->template getVariable<unsigned short>(FATALITY_DAYS);;
        unsigned short end_of_immunization_days = FLAMEGPU->template getVariable<unsigned short>(END_OF_IMMUNIZATION_DAYS);
        unsigned char infected_contact = FLAMEGPU->template getVariable<unsigned char>(INFECTED_CONTACT);

        auto coord2index = FLAMEGPU->environment.template getMacroProperty<short, FLOORS, ENV_DIM_Z, ENV_DIM_X>(COORD2INDEX);

        if(disease_state == SUSCEPTIBLE){
            // Contagion through contact
            float contamination_risk = FLAMEGPU->environment.template getProperty<float, RISK_CLASSES + 1>(CONTAMINATION_RISK, risk_class);

            const float contamination_risk_decreased_with_mask = FLAMEGPU->environment.template getProperty<float>(CONTAMINATION_RISK_DECREASED_WITH_MASK);
            const float virus_variant_factor = FLAMEGPU->environment.template getProperty<float>(VIRUS_VARIANT_FACTOR);
            const float area_around_agent = M_PI * RADIUS * RADIUS;

            const unsigned char mask_type = FLAMEGPU->template getVariable<unsigned char>(MASK_TYPE);

            contamination_risk = (mask_type != NO_MASK) ? contamination_risk * (1 - contamination_risk_decreased_with_mask): contamination_risk;

            // 0.01 because for now we are not considering symptomatic infected agents (otherwise
            // it would be 1/2^t where t is the time since the beginning of the symptoms)
            const float p_contact = infected_contact > 0 ? ((contamination_risk * 0.01) / area_around_agent): 0.0f;

            // Contagion through aerosol
            const float risk_const = FLAMEGPU->environment.template getProperty<float, RISK_CLASSES + 1>(RISK_CONST, risk_class);
            const float quanta_inhaled = FLAMEGPU->template getVariable<float>(QUANTA_INHALED);
            const float p_aerosol = 1 - exp(-(quanta_inhaled / risk_const));

            const float p = p_contact + p_aerosol - (p_contact * p_aerosol);

            const float random = cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_UNIFORM_0_1_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX)], UNIFORM, contacts_id, 0.0f, 1.0f, false);

            // See if the agent get the virus
            if(random < p){
#ifdef INCUBATION
                disease_state = EXPOSED;

                incubation_days = (unsigned short) round(cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_INCUBATION_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX)], FLAMEGPU->environment.template getProperty<unsigned short, RISK_CLASSES + 1>(MEAN_INCUBATION_DAYS_DISTR, risk_class), contacts_id, (float) FLAMEGPU->environment.template getProperty<unsigned short, RISK_CLASSES * 2 + 1>(MEAN_INCUBATION_DAYS, risk_class * 2), (float) FLAMEGPU->environment.template getProperty<unsigned short, RISK_CLASSES * 2 + 1>(MEAN_INCUBATION_DAYS, risk_class * 2 + 1), true));
#else
                disease_state = INFECTED;
                infection_days = (unsigned short) round(cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_INFECTION_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX)], FLAMEGPU->environment.template getProperty<unsigned short, RISK_CLASSES + 1>(MEAN_INFECTION_DAYS_DISTR, risk_class), contacts_id, (float) FLAMEGPU->environment.template getProperty<unsigned short, RISK_CLASSES * 2 + 1>(MEAN_INFECTION_DAYS, risk_class * 2), (float) FLAMEGPU->environment.template getProperty<unsigned short, RISK_CLASSES * 2 + 1>(MEAN_INFECTION_DAYS, risk_class * 2 + 1), true));
#ifdef FATALITY
                fatality_days = (unsigned short) round(cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_FATALITY_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX)], FLAMEGPU->environment.template getProperty<unsigned short, RISK_CLASSES + 1>(MEAN_FATALITY_DAYS_DISTR, risk_class), contacts_id, (float) FLAMEGPU->environment.template getProperty<unsigned short, RISK_CLASSES * 2 + 1>(MEAN_FATALITY_DAYS, risk_class * 2), (float) FLAMEGPU->environment.template getProperty<unsigned short, RISK_CLASSES * 2 + 1>(MEAN_FATALITY_DAYS, risk_class * 2 + 1), true));
#endif
#endif
                float agent_pos[3] = {FLAMEGPU->template getVariable<float>(X), FLAMEGPU->template getVariable<float>(Y), FLAMEGPU->template getVariable<float>(Z)};

                if(agent_pos[1] == INVISIBLE_AGENT_Y)
                    printf("0,%d,%d,%d,%d,%f,%f,%f,%d,-1\n", FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->template getVariable<short>(CONTACTS_ID), FLAMEGPU->template getVariable<short>(AGENT_TYPE), agent_pos[0], INVISIBLE_AGENT_Y, agent_pos[2], FLAMEGPU->template getVariable<unsigned char>(DISEASE_STATE));
                else
                    printf("0,%d,%d,%d,%d,%f,%f,%f,%d,%d\n", FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->template getVariable<short>(CONTACTS_ID), FLAMEGPU->template getVariable<short>(AGENT_TYPE), agent_pos[0], agent_pos[1], agent_pos[2], FLAMEGPU->template getVariable<unsigned char>(DISEASE_STATE), (short) coord2index[(unsigned short)(agent_pos[1]/YOFFSET)][(unsigned short)agent_pos[2]][(unsigned short)agent_pos[0]]);
            }
        }

        FLAMEGPU->template setVariable<float>(QUANTA_INHALED, 0.0f);
        FLAMEGPU->template setVariable<unsigned char>(INFECTED_CONTACT, 0);

        FLAMEGPU->template setVariable<unsigned char>(DISEASE_STATE, disease_state);
        FLAMEGPU->template setVariable<unsigned short>(INCUBATION_DAYS, incubation_days);
        FLAMEGPU->template setVariable<unsigned short>(INFECTION_DAYS, infection_days);
        FLAMEGPU->template setVariable<unsigned short>(FATALITY_DAYS, fatality_days);
        FLAMEGPU->template setVariable<unsigned short>(END_OF_IMMUNIZATION_DAYS, end_of_immunization_days);

#if defined(DEBUG) && !defined(ENSEMBLE)
        printf("5,%d,%d,Ending contagion_processes for agent with id %d\n", FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->template getVariable<short>(CONTACTS_ID));
#endif
        return ALIVE;
    }

/**
     * Update disease state.
    */
    template<typename MessageIn, typename MessageOut>
    FLAMEGPU_DEVICE_FUNCTION flamegpu::AGENT_STATUS update_infection(DeviceAPI<MessageIn, MessageOut> *FLAMEGPU){
#if defined(DEBUG) && !defined(ENSEMBLE)
        printf("5,%d,%d,Beginning of update_infected for agent with id %d\n", FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->template getVariable<short>(CONTACTS_ID));
#endif
        const short contacts_id = FLAMEGPU->template getVariable<short>(CONTACTS_ID);
        const unsigned char risk_class = FLAMEGPU->template getVariable<unsigned char>(RISK_CLASS);

        int disease_state = FLAMEGPU->template getVariable<unsigned char>(DISEASE_STATE);
        int disease_state_old = disease_state;
        unsigned short incubation_days = FLAMEGPU->template getVariable<unsigned short>(INCUBATION_DAYS);
        unsigned short infection_days = FLAMEGPU->template getVariable<unsigned short>(INFECTION_DAYS);
        unsigned short fatality_days = FLAMEGPU->template getVariable<unsigned short>(FATALITY_DAYS);
        unsigned short end_of_immunization_days = FLAMEGPU->template getVariable<unsigned short>(END_OF_IMMUNIZATION_DAYS);

        auto coord2index = FLAMEGPU->environment.template getMacroProperty<short, FLOORS, ENV_DIM_Z, ENV_DIM_X>(COORD2INDEX);

#ifdef INCUBATION
        if(disease_state == EXPOSED){
            if(!incubation_days){
                disease_state = INFECTED;

                infection_days = (unsigned short) round(cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_INFECTION_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX)], FLAMEGPU->environment.template getProperty<unsigned short, RISK_CLASSES + 1>(MEAN_INFECTION_DAYS_DISTR, risk_class), contacts_id, (float) FLAMEGPU->environment.template getProperty<unsigned short, RISK_CLASSES * 2 + 1>(MEAN_INFECTION_DAYS, risk_class * 2), (float) FLAMEGPU->environment.template getProperty<unsigned short, RISK_CLASSES * 2 + 1>(MEAN_INFECTION_DAYS, risk_class * 2 + 1), true));
#ifdef FATALITY
                fatality_days =  (unsigned short) round(cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_FATALITY_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX)], FLAMEGPU->environment.template getProperty<unsigned short, RISK_CLASSES + 1>(MEAN_FATALITY_DAYS_DISTR, risk_class), contacts_id, (float) FLAMEGPU->environment.template getProperty<unsigned short, RISK_CLASSES * 2 + 1>(MEAN_FATALITY_DAYS, risk_class * 2), (float) FLAMEGPU->environment.template getProperty<unsigned short, RISK_CLASSES * 2 + 1>(MEAN_FATALITY_DAYS, risk_class * 2 + 1), true));
#endif
            }
            else{
                incubation_days--;
            }
        }
#endif

        if(disease_state == INFECTED){
#ifdef FATALITY
            if(!fatality_days){
                disease_state = DIED;
#if defined(DEBUG) && !defined(ENSEMBLE)
                printf("5,%d,%d,Ending update_infected for agent with id %d\n", FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->template getVariable<short>(CONTACTS_ID));
#endif
                return DEAD;
            }
            else{
                fatality_days--;
            }
#endif

            if(!infection_days){
                disease_state = RECOVERED;
#ifdef REINFECTION
                end_of_immunization_days = (unsigned short) round(cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_END_OF_IMMUNIZATION_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX)], FLAMEGPU->environment.template getProperty<unsigned short, RISK_CLASSES + 1>(MEAN_END_OF_IMMUNIZATION_DAYS_DISTR, risk_class), contacts_id, (float) FLAMEGPU->environment.template getProperty<unsigned short, RISK_CLASSES * 2 + 1>(MEAN_END_OF_IMMUNIZATION_DAYS, risk_class * 2), (float) FLAMEGPU->environment.template getProperty<unsigned short, RISK_CLASSES * 2 + 1>(MEAN_END_OF_IMMUNIZATION_DAYS, risk_class * 2 + 1), true));
#endif
            }
            else{
                infection_days--;
            }
        }

#ifdef REINFECTION
        if(disease_state == RECOVERED){
            if(!end_of_immunization_days){
                disease_state = SUSCEPTIBLE;
            }
            else{
                end_of_immunization_days--;
            }
        }
#endif

        FLAMEGPU->template setVariable<unsigned char>(DISEASE_STATE, disease_state);
        FLAMEGPU->template setVariable<unsigned short>(INCUBATION_DAYS, incubation_days);
        FLAMEGPU->template setVariable<unsigned short>(INFECTION_DAYS, infection_days);
        FLAMEGPU->template setVariable<unsigned short>(FATALITY_DAYS, fatality_days);
        FLAMEGPU->template setVariable<unsigned short>(END_OF_IMMUNIZATION_DAYS, end_of_immunization_days);

        if(disease_state != disease_state_old){
            float agent_pos[3] = {FLAMEGPU->template getVariable<float>(X), FLAMEGPU->template getVariable<float>(Y), FLAMEGPU->template getVariable<float>(Z)};

            if(agent_pos[1] == INVISIBLE_AGENT_Y)
                printf("0,%d,%d,%d,%d,%f,%f,%f,%d,-1\n", FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->template getVariable<short>(CONTACTS_ID), FLAMEGPU->template getVariable<short>(AGENT_TYPE), agent_pos[0], INVISIBLE_AGENT_Y, agent_pos[2], FLAMEGPU->template getVariable<unsigned char>(DISEASE_STATE));
            else
                printf("0,%d,%d,%d,%d,%f,%f,%f,%d,%d\n", FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->template getVariable<short>(CONTACTS_ID), FLAMEGPU->template getVariable<short>(AGENT_TYPE), agent_pos[0], agent_pos[1], agent_pos[2], FLAMEGPU->template getVariable<unsigned char>(DISEASE_STATE), (short) coord2index[(unsigned short)(agent_pos[1]/YOFFSET)][(unsigned short)agent_pos[2]][(unsigned short)agent_pos[0]]);
        }
#if defined(DEBUG) && !defined(ENSEMBLE)
        printf("5,%d,%d,Ending update_infected for agent with id %d\n", FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->template getVariable<short>(CONTACTS_ID));
#endif
        return ALIVE;
    }

    /**
     * Handle outside contagion.
    */
    template<typename MessageIn, typename MessageOut>
    FLAMEGPU_DEVICE_FUNCTION void outside_contagion(DeviceAPI<MessageIn, MessageOut>* FLAMEGPU){
#if defined(DEBUG) && !defined(ENSEMBLE)
        printf("5,%d,%d,Beginning of outside_contagion for agent with id %d\n", FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->template getVariable<short>(CONTACTS_ID));
#endif
        if(FLAMEGPU->template getVariable<unsigned char>(DISEASE_STATE) == SUSCEPTIBLE && FLAMEGPU->template getVariable<unsigned char>(EXITED_FROM_ENVIRONMENT) && FLAMEGPU->template getVariable<float>(Y) == INVISIBLE_AGENT_Y){
            const short contacts_id = FLAMEGPU->template getVariable<short>(CONTACTS_ID);
            const unsigned short day = FLAMEGPU->environment.template getProperty<unsigned short>(DAY)-1;
            const float perc_inf = FLAMEGPU->environment.template getProperty<float, DAYS + 1>(PERC_INF, day);
            const unsigned char risk_class = FLAMEGPU->template getVariable<unsigned char>(RISK_CLASS);

            auto coord2index = FLAMEGPU->environment.template getMacroProperty<short, FLOORS, ENV_DIM_Z, ENV_DIM_X>(COORD2INDEX);

            float random = cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_UNIFORM_0_1_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX)], UNIFORM, contacts_id, 0.0f, 1.0f, false);
            if(random < perc_inf){
                auto counters = FLAMEGPU->environment.template getMacroProperty<unsigned int, NUM_COUNTERS>(COUNTERS);

                counters[NUM_INFECTED_OUTSIDE]++;

                if(FLAMEGPU->template getVariable<unsigned char>(AGENT_WITH_A_RATE) == AGENT_WITHOUT_RATE){
#ifdef INCUBATION
                    FLAMEGPU->template setVariable<unsigned char>(DISEASE_STATE, EXPOSED);

                    unsigned short incubation_days = (unsigned short) round(cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_INCUBATION_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX)], FLAMEGPU->environment.template getProperty<unsigned short, RISK_CLASSES + 1>(MEAN_INCUBATION_DAYS_DISTR, risk_class), contacts_id, (float) FLAMEGPU->environment.template getProperty<unsigned short, RISK_CLASSES * 2 + 1>(MEAN_INCUBATION_DAYS, risk_class * 2), (float) FLAMEGPU->environment.template getProperty<unsigned short, RISK_CLASSES * 2 + 1>(MEAN_INCUBATION_DAYS, risk_class * 2 + 1), true));
                    FLAMEGPU->template setVariable<unsigned short>(INCUBATION_DAYS, incubation_days);
#else
                    FLAMEGPU->template setVariable<unsigned char>(DISEASE_STATE, INFECTED);
                    unsigned short infection_days = (unsigned short) round(cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_INFECTION_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX)], FLAMEGPU->environment.template getProperty<unsigned short, RISK_CLASSES + 1>(MEAN_INFECTION_DAYS_DISTR, risk_class), contacts_id, (float) FLAMEGPU->environment.template getProperty<unsigned short, RISK_CLASSES * 2 + 1>(MEAN_INFECTION_DAYS, risk_class * 2), (float) FLAMEGPU->environment.template getProperty<unsigned short, RISK_CLASSES * 2>(MEAN_INFECTION_DAYS, risk_class * 2 + 1), true));
#endif

                }
                else{
                    FLAMEGPU->template setVariable<unsigned char>(DISEASE_STATE, INFECTED);
                    unsigned short infection_days = (unsigned short) round(cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_INFECTION_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX)], FLAMEGPU->environment.template getProperty<unsigned short, RISK_CLASSES + 1>(MEAN_INFECTION_DAYS_DISTR, risk_class), contacts_id, (float) FLAMEGPU->environment.template getProperty<unsigned short, RISK_CLASSES * 2 + 1>(MEAN_INFECTION_DAYS, risk_class * 2), (float) FLAMEGPU->environment.template getProperty<unsigned short, RISK_CLASSES * 2>(MEAN_INFECTION_DAYS, risk_class * 2 + 1), true));
                }

#ifdef FATALITY
                unsigned short fatality_days = (unsigned short) round(cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_FATALITY_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX)], FLAMEGPU->environment.template getProperty<unsigned short, RISK_CLASSES + 1>(MEAN_FATALITY_DAYS_DISTR, risk_class), contacts_id, (float) FLAMEGPU->environment.template getProperty<unsigned short, RISK_CLASSES * 2 + 1>(MEAN_FATALITY_DAYS, risk_class * 2), (float) FLAMEGPU->environment.template getProperty<unsigned short, RISK_CLASSES * 2 + 1>(MEAN_FATALITY_DAYS, risk_class * 2 + 1), true));
                FLAMEGPU->template setVariable<unsigned short>(FATALITY_DAYS, fatality_days);
#endif
                float agent_pos[3] = {FLAMEGPU->template getVariable<float>(X), FLAMEGPU->template getVariable<float>(Y), FLAMEGPU->template getVariable<float>(Z)};

                if(agent_pos[1] == INVISIBLE_AGENT_Y)
                    printf("0,%d,%d,%d,%d,%f,%f,%f,%d,-1\n", FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->template getVariable<short>(CONTACTS_ID), FLAMEGPU->template getVariable<short>(AGENT_TYPE), agent_pos[0], INVISIBLE_AGENT_Y, agent_pos[2], FLAMEGPU->template getVariable<unsigned char>(DISEASE_STATE));
                else
                    printf("0,%d,%d,%d,%d,%f,%f,%f,%d,%d\n", FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->template getVariable<short>(CONTACTS_ID), FLAMEGPU->template getVariable<short>(AGENT_TYPE), agent_pos[0], agent_pos[1], agent_pos[2], FLAMEGPU->template getVariable<unsigned char>(DISEASE_STATE), (short) coord2index[(unsigned short)(agent_pos[1]/YOFFSET)][(unsigned short)agent_pos[2]][(unsigned short)agent_pos[0]]);
            }
        }
#if defined(DEBUG) && !defined(ENSEMBLE)
        printf("5,%d,%d,Ending outside_contagion for agent with id %d\n", FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->template getVariable<short>(CONTACTS_ID));
#endif
    }

    /** 
     * Find the shortest path between two cells in matrix using A* algorithm.
    */
    template<typename MessageIn, typename MessageOut>
    FLAMEGPU_DEVICE_FUNCTION void a_star_matrix(DeviceAPI<MessageIn, MessageOut>* FLAMEGPU, const unsigned short room_id, const unsigned short start_pos[2], const unsigned short goal_pos[2], short* solution_x, short* solution_z) {
#if defined(DEBUG) && !defined(ENSEMBLE)
        printf("5,%d,%d,Beginning of a_star_matrix for agent with id %d\n", FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->template getVariable<short>(CONTACTS_ID));
#endif
        // ClosedSet: Matrix [z][x] storing parent coordinates encoded as (parent_z * MAX_DIMENSION + parent_x)
        short closedset[MAX_DIMENSION][MAX_DIMENSION];
        
        // OpenSet: Matrix [z][x][3] storing {F_Cost, G_Cost, Parent}
        short openset[MAX_DIMENSION][MAX_DIMENSION][3];

        // Initialization
        for (unsigned short z = 0; z < MAX_DIMENSION; ++z) {
            for (unsigned short x = 0; x < MAX_DIMENSION; ++x) {
                closedset[z][x] = NOT_PRESENT; 
                openset[z][x][F_COST] = NOT_PRESENT;
                openset[z][x][G_COST] = NOT_PRESENT;
                openset[z][x][PARENT] = NOT_PRESENT;
            }
        }

        // Extract coordinates from arrays
        unsigned short start_x = start_pos[0];
        unsigned short start_z = start_pos[1];
        unsigned short goal_x = goal_pos[0];
        unsigned short goal_z = goal_pos[1];

        // Get grid data from environment
        auto room_matrix = FLAMEGPU->environment.template getMacroProperty<short, V, MAX_DIMENSION, MAX_DIMENSION>(ROOM_MATRICES);

        // Get starting and goal cell values to determine which object cells agent can traverse
        short start_cell_value = (short) room_matrix[room_id][start_z][start_x];
        short goal_cell_value = (short) room_matrix[room_id][goal_z][goal_x];

        // Initialize start node
        short initial_h = CHEBYSHEV_DISTANCE(start_x, goal_x, start_z, goal_z);
        openset[start_z][start_x][F_COST] = initial_h;
        openset[start_z][start_x][G_COST] = 0;
        openset[start_z][start_x][PARENT] = STARTING_POINT;

        // Main A* loop
        for(unsigned short n_open = 1; n_open > 0;) {
            // 1. Find node with lowest F-cost
            short current_x = NOT_PRESENT;
            short current_z = NOT_PRESENT;
            short min_f = SHRT_MAX;

            for(unsigned short z = 0; z < MAX_DIMENSION; ++z) {
                for(unsigned short x = 0; x < MAX_DIMENSION; ++x) {
                    short f = openset[z][x][F_COST];
                    if (f != NOT_PRESENT && f < min_f) {
                        min_f = f;
                        current_x = x;
                        current_z = z;
                    }
                }
            }

            // Safety break
            if (current_x == NOT_PRESENT) break;

            // Get current node data
            short current_g = openset[current_z][current_x][G_COST];
            short current_parent = openset[current_z][current_x][PARENT];

            // 2. Move from OpenSet to ClosedSet
            openset[current_z][current_x][F_COST] = NOT_PRESENT;
            openset[current_z][current_x][G_COST] = NOT_PRESENT;
            openset[current_z][current_x][PARENT] = NOT_PRESENT;
            n_open--;

            closedset[current_z][current_x] = current_parent;

            // 3. Check if goal reached
            if (current_x == goal_x && current_z == goal_z) {
                // Backtrack to reconstruct full path as coordinates
                short full_path_x[SOLUTION_LENGTH];
                short full_path_z[SOLUTION_LENGTH];
                short length = 0;
                short curr_x = current_x;
                short curr_z = current_z;

                while ((curr_x != start_x || curr_z != start_z) && length < SOLUTION_LENGTH) {
                    full_path_x[length] = curr_x;
                    full_path_z[length] = curr_z;
                    length++;
                    short parent = closedset[curr_z][curr_x];
                    if (parent == STARTING_POINT) break;
                    curr_x = parent % MAX_DIMENSION;
                    curr_z = parent / MAX_DIMENSION;
                }

                // Add start point if not already there
                if (length < SOLUTION_LENGTH) {
                    full_path_x[length] = start_x;
                    full_path_z[length] = start_z;
                    length++;
                }

                // Reverse path from Start -> Goal
                short reversed_x[SOLUTION_LENGTH];
                short reversed_z[SOLUTION_LENGTH];
                for(unsigned short i = 0, j = length - 1; i < length; ++i, --j){
                    reversed_x[i] = full_path_x[j];
                    reversed_z[i] = full_path_z[j];
                }

                // Simplify: keep only waypoints where direction changes
                unsigned short wp_count = 0;
                // Start point at starting position
                solution_x[wp_count] = start_x;
                solution_z[wp_count] = start_z;
                wp_count++;

                // Check for direction changes starting from second point
                for(unsigned short i = 1; i < length && wp_count < SOLUTION_LENGTH / 2; ++i) {
                    short curr_x = reversed_x[i];
                    short curr_z = reversed_z[i];
                    short prev_x = reversed_x[i - 1];
                    short prev_z = reversed_z[i - 1];
                    
                    // Direction from previous waypoint to current point
                    short dir_x_to_curr = (curr_x > prev_x) ? 1 : (curr_x < prev_x) ? -1 : 0;
                    short dir_z_to_curr = (curr_z > prev_z) ? 1 : (curr_z < prev_z) ? -1 : 0;

                    // Check if next point requires direction change
                    bool is_waypoint = false;
                    if (i < length - 1) {
                        short next_x = reversed_x[i + 1];
                        short next_z = reversed_z[i + 1];
                        
                        short dir_x_to_next = (next_x > curr_x) ? 1 : (next_x < curr_x) ? -1 : 0;
                        short dir_z_to_next = (next_z > curr_z) ? 1 : (next_z < curr_z) ? -1 : 0;
                        
                        // It's a waypoint if direction changes
                        if (dir_x_to_curr != dir_x_to_next || dir_z_to_curr != dir_z_to_next) {
                            is_waypoint = true;
                        }
                    } else {
                        // Last point is always a waypoint (goal)
                        is_waypoint = true;
                    }

                    if (is_waypoint) {
                        // Store as relative coordinates from start
                        solution_x[wp_count] = curr_x;
                        solution_z[wp_count] = curr_z;
                        wp_count++;
                    }
                }

                // Line-of-sight optimization: skip intermediate waypoints if reachable directly
                unsigned short final_wp_count = 1; // Always keep start point
                for (unsigned short i = 1; i < wp_count; ++i) {
                    bool can_reach_directly = false;
                    short farthest_j = i;

                    // Try to reach as far as possible in direct line
                    for (short j = i + 1; j < wp_count; ++j) {
                        short x1 = solution_x[final_wp_count - 1];
                        short z1 = solution_z[final_wp_count - 1];
                        short x2 = solution_x[j];
                        short z2 = solution_z[j];

                        // Check all points along the line using Chebyshev distance
                        bool line_clear = true;
                        short dx = x2 - x1;
                        short dz = z2 - z1;
                        short abs_dx = (dx < 0) ? -dx : dx;
                        short abs_dz = (dz < 0) ? -dz : dz;
                        short steps = (abs_dx > abs_dz) ? abs_dx : abs_dz;

                        // Check each step along the line with proper scaling
                        for (short step = 0; step <= steps && line_clear; ++step) {
                            // Use integer division with rounding for each coordinate
                            short check_x = x1 + (steps > 0 ? (dx * step + steps / 2) / steps : 0);
                            short check_z = z1 + (steps > 0 ? (dz * step + steps / 2) / steps : 0);

                            short cell_val = (short) room_matrix[room_id][check_z][check_x];
                            if (cell_val == 0 || (cell_val != 1 && cell_val != 2 && cell_val != start_cell_value && cell_val != goal_cell_value)) {
                                line_clear = false;
                            }
                        }

                        if (line_clear) {
                            farthest_j = j;
                            can_reach_directly = true;
                        } else {
                            break;
                        }
                    }

                    // Add the farthest reachable waypoint
                    if (can_reach_directly) {
                        solution_x[final_wp_count] = solution_x[farthest_j];
                        solution_z[final_wp_count] = solution_z[farthest_j];
                        final_wp_count++;
                        i = farthest_j;
                    } else {
                        // Can't reach further, add current waypoint
                        solution_x[final_wp_count] = solution_x[i];
                        solution_z[final_wp_count] = solution_z[i];
                        final_wp_count++;
                    }
                }

                // Fill rest with -1
                for(unsigned short i = final_wp_count; i < SOLUTION_LENGTH; ++i) {
                    solution_x[i] = -1;
                    solution_z[i] = -1;
                }
#if defined(DEBUG) && !defined(ENSEMBLE)
                printf("5,%d,%d,Ending of a_star_matrix for agent with id %d\n", FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->template getVariable<short>(CONTACTS_ID));
#endif
                return;
            }

            // 4. Expand neighbors (4 directions)
            const short dirs_x[4] = {0, 0, -1, 1};
            const short dirs_z[4] = {-1, 1, 0, 0};

            for (int k = 0; k < 4; ++k) {
                short nx = current_x + dirs_x[k];
                short nz = current_z + dirs_z[k];

                if (nx < 0 || nx >= MAX_DIMENSION || nz < 0 || nz >= MAX_DIMENSION)
                    continue;

                // Check if neighbor cell is passable: walkable (1), door (2), or matches start cell value
                short neighbor_value = (short) room_matrix[room_id][nz][nx];
                if (neighbor_value == 0 || closedset[nz][nx] != NOT_PRESENT)
                    continue;
                
                // Cell is passable if: 1 (walkable), 2 (door), or equals start_cell_value (agent's current object)
                if (neighbor_value != 1 && neighbor_value != 2 && neighbor_value != start_cell_value && neighbor_value != goal_cell_value)
                    continue;

                bool is_diagonal = (dirs_x[k] != 0 && dirs_z[k] != 0);
                if (is_diagonal) {
                    short cell_x = (short) room_matrix[room_id][current_z][nx];
                    short cell_z = (short) room_matrix[room_id][nz][current_x];

                    // And must match passability criteria
                    if (((cell_x != 1 && cell_x != 2 && cell_x != start_cell_value && cell_x != goal_cell_value) ||
                         (cell_z != 1 && cell_z != 2 && cell_z != start_cell_value && cell_z != goal_cell_value)))
                        continue;
                }

                short new_g = current_g + 1;
                short old_g = openset[nz][nx][G_COST];
                bool is_in_open = (old_g != NOT_PRESENT);

                if (!is_in_open || new_g < old_g) {
                    short h = CHEBYSHEV_DISTANCE(nx, goal_x, nz, goal_z);
                    openset[nz][nx][F_COST] = new_g + h;
                    openset[nz][nx][G_COST] = new_g;
                    openset[nz][nx][PARENT] = current_z * MAX_DIMENSION + current_x;

                    if (!is_in_open)
                        n_open++;
                }
            }
        }
#if defined(DEBUG) && !defined(ENSEMBLE)
        printf("5,%d,%d,Ending of a_star_matrix for agent with id %d\n", FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->template getVariable<short>(CONTACTS_ID));
#endif
    }

/**
     * Update agent intermediate and final targets.
    */
    template<typename MessageIn, typename MessageOut>
    FLAMEGPU_DEVICE_FUNCTION void update_targets_coordinates(DeviceAPI<MessageIn, MessageOut>* FLAMEGPU, unsigned short actual_node, short* new_targets_x, short* new_targets_z, unsigned short *target_index, const int stay) {
#if defined(DEBUG) && !defined(ENSEMBLE)
        printf("5,%d,%d,Beginning of update_targets_coordinates for agent with id %d\n", FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->template getVariable<short>(CONTACTS_ID));
#endif
        auto intermediate_target_x = FLAMEGPU->environment.template getMacroProperty<float, TOTAL_AGENTS_ESTIMATION, SOLUTION_LENGTH>(INTERMEDIATE_TARGET_X);
        auto intermediate_target_y = FLAMEGPU->environment.template getMacroProperty<float, TOTAL_AGENTS_ESTIMATION, SOLUTION_LENGTH>(INTERMEDIATE_TARGET_Y);
        auto intermediate_target_z = FLAMEGPU->environment.template getMacroProperty<float, TOTAL_AGENTS_ESTIMATION, SOLUTION_LENGTH>(INTERMEDIATE_TARGET_Z);
        auto stay_matrix = FLAMEGPU->environment.template getMacroProperty<unsigned int, TOTAL_AGENTS_ESTIMATION, SOLUTION_LENGTH>(STAY);

        float agent_pos[3] = {FLAMEGPU->template getVariable<float>(X), FLAMEGPU->template getVariable<float>(Y), FLAMEGPU->template getVariable<float>(Z)};
        float new_target_x, new_target_y, new_target_z;

        const short contacts_id = FLAMEGPU->template getVariable<short>(CONTACTS_ID);

        short i = 1;
        while(i < SOLUTION_LENGTH && new_targets_x[i] != -1){
            float x = FLAMEGPU->environment.template getProperty<float, V>(NODE_X, new_targets_x[i]);
            float z = FLAMEGPU->environment.template getProperty<float, V>(NODE_Z, new_targets_z[i]);

            float offset_x = cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_UNIFORM_0_1_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX)], UNIFORM, contacts_id, 0.0f, 1.0f - 1e-3, false);
            float offset_z = cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_UNIFORM_0_1_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX)], UNIFORM, contacts_id, 0.0f, 1.0f - 1e-3, false);

            new_target_x = (float) new_targets_x[i] + FLAMEGPU->environment.template getProperty<float, V>(NODE_X, actual_node) - 1.0f + offset_x;
            new_target_y = agent_pos[1];
            new_target_z = (float) new_targets_z[i] + FLAMEGPU->environment.template getProperty<float, V>(NODE_Z, actual_node) - 1.0f + offset_z;

            intermediate_target_x[contacts_id][*target_index].exchange(new_target_x);
            intermediate_target_y[contacts_id][*target_index].exchange(new_target_y);
            intermediate_target_z[contacts_id][*target_index].exchange(new_target_z);

            *target_index = (*target_index + 1) % SOLUTION_LENGTH;

            ++i;
        }

        stay_matrix[contacts_id][*target_index].exchange(stay);
        FLAMEGPU->template setVariable<unsigned short>(TARGET_INDEX, *target_index);

#if defined(DEBUG) && !defined(ENSEMBLE)
        printf("5,%d,%d,Ending update_targets_coordinates for agent with id %d\n", FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->template getVariable<short>(CONTACTS_ID));
#endif
    }

    /** 
     * Handle room to door logic.
    */
    template<typename MessageIn, typename MessageOut>
    FLAMEGPU_DEVICE_FUNCTION void room2door_logic(DeviceAPI<MessageIn, MessageOut>* FLAMEGPU, short start_node){
#if defined(DEBUG) && !defined(ENSEMBLE)
        printf("5,%d,%d,Beginning of room2door_logic for agent with id %d\n", FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->template getVariable<short>(CONTACTS_ID));
#endif
        auto room_doors_position = FLAMEGPU->environment.template getMacroProperty<float, V, 2>(ROOM_DOORS_POSITION);
        
        float agent_pos[3] = {FLAMEGPU->template getVariable<float>(X), FLAMEGPU->template getVariable<float>(Y), FLAMEGPU->template getVariable<float>(Z)};
        unsigned short target_index = FLAMEGPU->template getVariable<unsigned short>(TARGET_INDEX);
        unsigned short start_position[2] = {(unsigned short) (agent_pos[0] - FLAMEGPU->environment.template getProperty<float, V>(NODE_X, start_node) + 1.0f), (unsigned short) (agent_pos[2] - FLAMEGPU->environment.template getProperty<float, V>(NODE_Z, start_node) + 1.0f)};
        unsigned short final_position[2] = {(unsigned short) room_doors_position[start_node][0], (unsigned short) room_doors_position[start_node][1]};
        short solution_x[SOLUTION_LENGTH] = {-1};
        short solution_z[SOLUTION_LENGTH] = {-1};


        a_star_matrix(FLAMEGPU, start_node, start_position, final_position, solution_x, solution_z);
        update_targets_coordinates(FLAMEGPU, start_node, solution_x, solution_z, &target_index, 0);

        FLAMEGPU->template setVariable<unsigned char>(MOVEMENT_PHASE, ROOM2DOOR);
#if defined(DEBUG) && !defined(ENSEMBLE)
        printf("5,%d,%d,Ending of room2door_logic for agent with id %d\n", FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->template getVariable<short>(CONTACTS_ID));
#endif
    }

    /** 
     * Handle room to room logic (e.g. objects/obstacles, objects resources, path finding, etc.)
    */
    template<typename MessageIn, typename MessageOut>
    FLAMEGPU_DEVICE_FUNCTION void room2room_logic(DeviceAPI<MessageIn, MessageOut>* FLAMEGPU){
#if defined(DEBUG) && !defined(ENSEMBLE)
        printf("5,%d,%d,Beginning of room2room_logic for agent with id %d\n", FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->template getVariable<short>(CONTACTS_ID));
#endif
        auto coord2index = FLAMEGPU->environment.template getMacroProperty<short, FLOORS, ENV_DIM_Z, ENV_DIM_X>(COORD2INDEX);
        auto room_doors_position = FLAMEGPU->environment.template getMacroProperty<float, V, 2>(ROOM_DOORS_POSITION);

        unsigned short target_index = FLAMEGPU->template getVariable<unsigned short>(TARGET_INDEX);
        float agent_pos[3] = {FLAMEGPU->template getVariable<float>(X), FLAMEGPU->template getVariable<float>(Y), FLAMEGPU->template getVariable<float>(Z)};
        short start_node = coord2index[(unsigned short)(agent_pos[1]/YOFFSET)][(unsigned short)agent_pos[2]][(unsigned short)agent_pos[0]];
        short actual_node = FLAMEGPU->template getVariable<short>(ACTUAL_NODE);
        short final_node;
        short solution[SOLUTION_LENGTH] = {-1};
        unsigned short final_door_position_x, final_door_position_y, final_door_position_z;

        printf("[TEMP_DEBUG] start_node: %d, actual_node: %d\n", start_node, actual_node);
        printf("[TEMP_DEBUG] agent_pos: (%f, %f, %f)\n", agent_pos[0], agent_pos[1], agent_pos[2]);
        
        printf("[TEMP_DEBUG] room_doors_position[actual_node]: (%f, %f)\n", (float) room_doors_position[actual_node][0], (float) room_doors_position[actual_node][1]);
        printf("[TEMP_DEBUG] NODE_X: %f, NODE_Z: %f\n", FLAMEGPU->environment.template getProperty<float, V>(NODE_X, actual_node), FLAMEGPU->environment.template getProperty<float, V>(NODE_Z, actual_node));

        // Print NODE_X and NODE_Z for all nodes
        for(short node = 0; node < V; ++node) {
            printf("[TEMP_DEBUG] NODE_X[%d]: %f, NODE_Z[%d]: %f\n", node, FLAMEGPU->environment.template getProperty<float, V>(NODE_X, node), node, FLAMEGPU->environment.template getProperty<float, V>(NODE_Z, node));
        }

        for(short node = 0; node < V; ++node) {
            printf("[TEMP_DEBUG] INDEX2COORD[%d]: (%d, %d, %d)\n", node, (unsigned short) FLAMEGPU->environment.template getProperty<unsigned short, V>(INDEX2COORDX, node), (unsigned short) FLAMEGPU->environment.template getProperty<unsigned short, V>(INDEX2COORDY, node), (unsigned short) FLAMEGPU->environment.template getProperty<unsigned short, V>(INDEX2COORDZ, node));
        }

        // Print room doors position for all nodes
        for(short node = 0; node < V; ++node) {
            printf("[TEMP_DEBUG] room_doors_position[%d]: (%f, %f)\n", node, (float) room_doors_position[node][0], (float) room_doors_position[node][1]);
        }

        final_door_position_x = (unsigned short) (((float) room_doors_position[actual_node][0]) + FLAMEGPU->environment.template getProperty<float, V>(NODE_X, actual_node) - 1.0f);
        final_door_position_y = FLAMEGPU->environment.template getProperty<unsigned short, V>(INDEX2COORDY, actual_node);
        final_door_position_z = (unsigned short) (((float) room_doors_position[actual_node][1]) + FLAMEGPU->environment.template getProperty<float, V>(NODE_Z, actual_node) - 1.0f);

        printf("[TEMP_DEBUG] final_door_position: (%d, %d, %d)\n", final_door_position_x, final_door_position_y, final_door_position_z);

        final_node = coord2index[final_door_position_y/YOFFSET][final_door_position_z][final_door_position_x];

        printf("[TEMP_DEBUG] final_node: %d\n", final_node);

        a_star(FLAMEGPU, start_node, final_node, solution);
        update_targets(FLAMEGPU, solution, &target_index, false, 0);

        FLAMEGPU->template setVariable<unsigned char>(MOVEMENT_PHASE, ROOM2ROOM);
#if defined(DEBUG) && !defined(ENSEMBLE)
        printf("5,%d,%d,Ending of room2room_logic for agent with id %d\n", FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->template getVariable<short>(CONTACTS_ID));
#endif
    }

    /** 
     * Handle door to room logic (e.g. objects/obstacles, objects resources, path finding, etc.)
    */
    template<typename MessageIn, typename MessageOut>
    FLAMEGPU_DEVICE_FUNCTION void door2room_logic(DeviceAPI<MessageIn, MessageOut>* FLAMEGPU){
#if defined(DEBUG) && !defined(ENSEMBLE)
        printf("5,%d,%d,Beginning of door2room_logic for agent with id %d\n", FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->template getVariable<short>(CONTACTS_ID));
#endif
        auto rooms_has_objects = FLAMEGPU->environment.template getMacroProperty<short, V>(ROOMS_HAS_OBJECTS);
        auto coord2index = FLAMEGPU->environment.template getMacroProperty<short, FLOORS, ENV_DIM_Z, ENV_DIM_X>(COORD2INDEX);

        const short contacts_id = FLAMEGPU->template getVariable<short>(CONTACTS_ID);

        float agent_pos[3] = {FLAMEGPU->template getVariable<float>(X), FLAMEGPU->template getVariable<float>(Y), FLAMEGPU->template getVariable<float>(Z)};
        short start_node = coord2index[(unsigned short)(agent_pos[1]/YOFFSET)][(unsigned short)agent_pos[2]][(unsigned short)agent_pos[0]];
        short actual_node = FLAMEGPU->template getVariable<short>(ACTUAL_NODE);
        short actual_node_stay = FLAMEGPU->template getVariable<short>(ACTUAL_NODE_STAY);
        short agent_type = FLAMEGPU->template getVariable<short>(AGENT_TYPE);
        short solution_x[SOLUTION_LENGTH] = {-1};
        short solution_z[SOLUTION_LENGTH] = {-1};
        short selected_object = -1;
        unsigned short target_index = FLAMEGPU->template getVariable<unsigned short>(TARGET_INDEX);
        unsigned short start_position[2] = {(unsigned short) (agent_pos[0] - FLAMEGPU->environment.template getProperty<float, V>(NODE_X, start_node) - 1.0f), (unsigned short) (agent_pos[2] - FLAMEGPU->environment.template getProperty<float>(NODE_Z, start_node) - 1.0f)};
        unsigned short final_position[2] = {0};

        if((char) rooms_has_objects[actual_node]){
            auto rooms_resources_global_objects = FLAMEGPU->environment.template getMacroProperty<unsigned int, V, MAX_OBJECTS+1>(ROOMS_RESOURCES_GLOBAL_OBJECTS);
            auto rooms_resources_global_objects_counter = FLAMEGPU->environment.template getMacroProperty<unsigned int, V, MAX_OBJECTS+1>(ROOMS_RESOURCES_GLOBAL_OBJECTS_COUNTER);
            auto rooms_resources_specific_objects = FLAMEGPU->environment.template getMacroProperty<unsigned int, NUMBER_OF_AGENTS_TYPES, V, MAX_OBJECTS+1>(ROOMS_RESOURCES_SPECIFIC_OBJECTS);
            auto rooms_resources_specific_objects_counter = FLAMEGPU->environment.template getMacroProperty<unsigned int, NUMBER_OF_AGENTS_TYPES, V, MAX_OBJECTS+1>(ROOMS_RESOURCES_SPECIFIC_OBJECTS_COUNTER);

            // Random policy
            unsigned short random_object = (unsigned short) cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_UNIFORM_0_1_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX)], UNIFORM, contacts_id, 0.0f, (float) (MAX_OBJECTS-1), false);
            unsigned short random_object_start = random_object;

            do{
                unsigned int get_specific_resources = rooms_resources_specific_objects_counter[agent_type][actual_node][random_object]++;

                if(get_specific_resources <= rooms_resources_specific_objects[agent_type][actual_node][random_object]){
                    unsigned int get_global_resources = rooms_resources_global_objects_counter[agent_type][actual_node][random_object]++;
                    if(get_global_resources <= rooms_resources_global_objects[agent_type][actual_node][random_object]){
                        selected_object = random_object;
                    }
                    else{
                        rooms_resources_global_objects_counter[agent_type][actual_node][random_object]--;
                        random_object = (random_object + 1) % MAX_OBJECTS;
                    }
                }
                else{
                    rooms_resources_specific_objects_counter[agent_type][actual_node][random_object]--;
                    random_object = (random_object + 1) % MAX_OBJECTS;
                }
            } while(selected_object == -1 && random_object != random_object_start);

            // Other policies
            // TO DO

            // No resource available for the interested objects, stay standing in a random position with a 1 in the matrix.
            // To implement: probability to stay standing even if there are available resources.
            if(selected_object == -1){
                FLAMEGPU->template setVariable<short>(ACTUAL_NODE_OBJECT, -1);

                auto room_matrix = FLAMEGPU->environment.template getMacroProperty<short, V, MAX_DIMENSION, MAX_DIMENSION>(ROOM_MATRICES);

                // TO DO: CONSIDEARE LA ROTAZIONE DELLA STANZA! VIENE FATTO GIA' DA F4F? Anche delle coordinate e dimensioni degli oggetti nella stanza?

                // Select a position (x, z) in the matrix with a 1 at random.
                unsigned short room_length = (unsigned short) FLAMEGPU->environment.template getProperty<float>(NODE_LENGTH, actual_node);
                unsigned short room_width = (unsigned short) FLAMEGPU->environment.template getProperty<float>(NODE_WIDTH, actual_node);
                unsigned short positions[MAX_DIMENSION * MAX_DIMENSION][2];
                unsigned short num_positions = 0;
                for(unsigned short z = 1; z < room_width; ++z){
                    for(unsigned short x = 1; x < room_length; ++x){
                        if((short) room_matrix[actual_node][z][x] == 1){
                            if(num_positions < MAX_DIMENSION * MAX_DIMENSION){
                                positions[num_positions][0] = x;
                                positions[num_positions][1] = z;
                                num_positions++;
                            }
                        }
                    }
                }

                if(num_positions > 0){
                    unsigned short random_index = (unsigned short) round(cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_UNIFORM_0_1_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX)], UNIFORM, FLAMEGPU->template getVariable<short>(CONTACTS_ID), 0.0f, (float) num_positions - 1.0f, false));
                    final_position[0] = positions[random_index][0] + 1;
                    final_position[1] = positions[random_index][1] + 1;
                }
            }
            else{
                auto rooms_x_objects = FLAMEGPU->environment.template getMacroProperty<float, V, MAX_OBJECTS+1>(ROOMS_X_OBJECTS);
                auto rooms_z_objects = FLAMEGPU->environment.template getMacroProperty<float, V, MAX_OBJECTS+1>(ROOMS_Z_OBJECTS);
                auto rooms_length_objects = FLAMEGPU->environment.template getMacroProperty<float, V, MAX_OBJECTS+1>(ROOMS_LENGTH_OBJECTS);
                auto rooms_width_objects = FLAMEGPU->environment.template getMacroProperty<float, V, MAX_OBJECTS+1>(ROOMS_WIDTH_OBJECTS);

                unsigned short x_object = (unsigned short) rooms_x_objects[actual_node][selected_object];
                unsigned short z_object = (unsigned short) rooms_z_objects[actual_node][selected_object];
                unsigned short length_object = (unsigned short) rooms_length_objects[actual_node][selected_object];
                unsigned short width_object = (unsigned short) rooms_width_objects[actual_node][selected_object];

                unsigned short random_offset_x = (unsigned short) round(cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_UNIFORM_0_1_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX)], UNIFORM, FLAMEGPU->template getVariable<short>(CONTACTS_ID), 0.0f, length_object - 1e-3, false));
                unsigned short random_offset_z = (unsigned short) round(cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_UNIFORM_0_1_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX)], UNIFORM, FLAMEGPU->template getVariable<short>(CONTACTS_ID), 0.0f, width_object - 1e-3, false));

                final_position[0] = (x_object + random_offset_x) - FLAMEGPU->environment.template getProperty<float, V>(NODE_X, start_node) + 1.0f;
                final_position[1] = (z_object + random_offset_z) - FLAMEGPU->environment.template getProperty<float, V>(NODE_Z, start_node) + 1.0f;
            }
        }
        else{
            float jitter_x = 0.0f;
            float jitter_z = 0.0f;

            // Generate a random offset
            generate_offset(FLAMEGPU, &jitter_x, &jitter_z, actual_node);
            final_position[0] = jitter_x + 1;
            final_position[1] = jitter_z + 1;
        }

        a_star_matrix(FLAMEGPU, start_node, start_position, final_position, solution_x, solution_z);
        update_targets_coordinates(FLAMEGPU, actual_node, solution_x, solution_z, &target_index, actual_node_stay);

        // Save the index of the object selected by the agent to update the resources availability when the agent will exit the room.
        FLAMEGPU->template setVariable<short>(ACTUAL_NODE_OBJECT, selected_object);

        FLAMEGPU->template setVariable<unsigned char>(MOVEMENT_PHASE, DOOR2ROOM);
#if defined(DEBUG) && !defined(ENSEMBLE)
        printf("5,%d,%d,Ending of door2room_logic for agent with id %d\n", FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->template getVariable<short>(CONTACTS_ID));
#endif
    }

    FLAMEGPU_DEVICE_FUNCTION unsigned char findLeftmostIndex(const double target, const double *env_events_cdf, const short num_events) {
        int left = 0;
        int right = num_events - 1;

        if (target > env_events_cdf[1])
            return left;

        if (target <= env_events_cdf[right])
            return right;

        while (left <= right) {
            int mid = left + (right - left) / 2;

            double upper = env_events_cdf[mid];
            double lower = env_events_cdf[mid + 1];

            if (target <= upper && target > lower) {
                return mid;
            } else if (target > upper) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }

        return left;
    }
}
#endif //_DEVICE_FUNCTIONS_CUH_
