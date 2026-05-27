#ifndef _DEVICE_FUNCTIONS_CUH_
#define _DEVICE_FUNCTIONS_CUH_

#include "defines.h"

using namespace flamegpu;
using namespace std;

namespace device_functions {
    /**
     * Compare floating points.
    */
    FLAMEGPU_DEVICE_FUNCTION bool compare_float(const double a, const double b, const double epsilon) {
        return fabs(a - b) < epsilon;
    }

    /**
     * Generate a random number using the given RNG, distribution and parameters for pedestrians.
    */
    template<typename MessageIn, typename MessageOut>
    FLAMEGPU_DEVICE_FUNCTION float cuda_pedestrian_rng(DeviceAPI<MessageIn, MessageOut>* FLAMEGPU, unsigned short distribution_id, curandState *cuda_states, int type, short id, float a, float b, bool flow_time) {
        float random = (type == TRUNCATED_POSITIVE_NORMAL) ? curand_normal(&cuda_states[id]): curand_uniform(&cuda_states[id]);

        if(type == EXPONENTIAL && compare_float((double) random, 1.0f, 1e-10f)){
            do{
                random = curand_uniform(&cuda_states[id]);
            }while(compare_float((double) random, 1.0f, 1e-10f));
        }
        const float event_time_random = DISTRIBUTION(type, random, a, b);

        auto cuda_rng_offsets_pedestrian = FLAMEGPU->environment.template getMacroProperty<unsigned int, TOTAL_AGENTS_ESTIMATION>(CUDA_RNG_OFFSETS_PEDESTRIAN);
        cuda_rng_offsets_pedestrian[FLAMEGPU->template getVariable<short>(CONTACTS_ID)]++;

        return (flow_time && event_time_random < 1.0f) ? 1.0f: event_time_random;
    }

    /**
     * Generate a random number using the given RNG, distribution and parameters for rooms.
    */
    FLAMEGPU_DEVICE_FUNCTION float cuda_room_rng(DeviceAPI<MessageBucket, MessageBucket>* FLAMEGPU, unsigned short distribution_id, curandState *cuda_states, int type, short id, float a, int b, bool flow_time) {
        float random = (type == TRUNCATED_POSITIVE_NORMAL) ? curand_normal(&cuda_states[id]): curand_uniform(&cuda_states[id]);

        if(type == EXPONENTIAL && compare_float((double) random, 1.0f, 1e-10f)){
            do{
                random = curand_uniform(&cuda_states[id]);
            }while(compare_float((double) random, 1.0f, 1e-10f));
        }
        const float event_time_random = DISTRIBUTION(type, random, a, b);

        auto cuda_rng_offsets_room = FLAMEGPU->environment.getMacroProperty<unsigned int, NUM_ROOMS>(CUDA_RNG_OFFSETS_ROOM);
        cuda_rng_offsets_room[FLAMEGPU->getID()]++;

        return (flow_time && event_time_random < 1.0f) ? 1.0f: event_time_random;
    }

    /**
     * Generate a random offset inside rooms.
    */
    FLAMEGPU_DEVICE_FUNCTION void generate_offset(DeviceAPI<MessageBucket, MessageBucket>* FLAMEGPU, float* jitter_x, float* jitter_z,  short new_target){
#if defined(DEBUG) && !defined(ENSEMBLE)
        printf("5,%d,%d,Beginning of generate_offset for agent with id %d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID));
#endif
        const short contacts_id = FLAMEGPU->getVariable<short>(CONTACTS_ID);
        const float yaw = FLAMEGPU->environment.getProperty<float, V>(NODE_YAW, new_target);
        const bool yaw_condition = compare_float(yaw, M_PI/2, 0.5f) || compare_float(yaw, 2*M_PI - M_PI/2, 0.5f);

        float length = FLAMEGPU->environment.getProperty<float, V>(NODE_LENGTH, new_target);
        float width = FLAMEGPU->environment.getProperty<float, V>(NODE_WIDTH, new_target);
        float offset_x, offset_z;

        offset_x = yaw_condition ? width: length;
        offset_z = yaw_condition ? length: width;

        *jitter_x = cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_JITTER_X_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX)], UNIFORM, contacts_id, 0.0f, offset_x - 1e-3, false);
        *jitter_z = cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_JITTER_Z_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX)], UNIFORM, contacts_id, 0.0f, offset_z - 1e-3, false);
#if defined(DEBUG) && !defined(ENSEMBLE)
        printf("5,%d,%d,Ending of generate_offset for agent with id %d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID));
#endif
    }

 /**
     * Find a room of free resources for an event, searching the nearest
    */
    FLAMEGPU_DEVICE_FUNCTION short findFreeRoomForEventOfTypeAndArea(DeviceAPI<MessageBucket, MessageBucket>* FLAMEGPU, float previous_separation, int type_room_event, int area_room_event, bool *available) {
#if defined(DEBUG) && !defined(ENSEMBLE)
        printf("5,%d,%d,Beginning of findFreeRoomForEventOfTypeAndArea for agent with id %d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID));
#endif
        const int agent_type = FLAMEGPU->getVariable<int>(AGENT_TYPE);
        short event_node;
        float min_separation = numeric_limits<float>::max();
        float agent_pos[3] = {FLAMEGPU->getVariable<float>(X), FLAMEGPU->getVariable<float>(Y), FLAMEGPU->getVariable<float>(Z)};
        //resources
        auto global_resources = FLAMEGPU->environment.getMacroProperty<int, V>(GLOBAL_RESOURCES);
        auto global_resources_counter = FLAMEGPU->environment.getMacroProperty<unsigned int, V>(GLOBAL_RESOURCES_COUNTER);
        auto specific_resources = FLAMEGPU->environment.getMacroProperty<int, NUMBER_OF_AGENTS_TYPES, V>(SPECIFIC_RESOURCES);
        auto specific_resources_counter = FLAMEGPU->environment.getMacroProperty<unsigned int, NUMBER_OF_AGENTS_TYPES, V>(SPECIFIC_RESOURCES_COUNTER);

        do {
            // Searching the nearest room related to the event
            //to add the area

            float min_separation = numeric_limits<float>::max();
            event_node = -1;
            for(const auto& message: FLAMEGPU->message_in(type_room_event)) {

                const unsigned short near_agent_pos[3] = {message.getVariable<unsigned short>(X), message.getVariable<unsigned short>(Y), message.getVariable<unsigned short>(Z)};
                int area_room = message.getVariable<int>(AREA);

                float separation = abs(near_agent_pos[0] - agent_pos[0]) + abs(near_agent_pos[1] - agent_pos[1]) + abs(near_agent_pos[2] - agent_pos[2]);
                if(separation < min_separation && separation > previous_separation && area_room_event == area_room){
                    min_separation = separation;
                    event_node = message.getVariable<short>(GRAPH_NODE);
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
        printf("5,%d,%d,Ending of findFreeRoomForEventOfTypeAndAre for agent with id %d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID));
#endif
        return event_node;
    }

    /**
     * Find a room of free resources
    */
    FLAMEGPU_DEVICE_FUNCTION short findFreeRoomOfTypeAndArea(DeviceAPI<MessageBucket, MessageBucket>* FLAMEGPU, int flow, int random, int lenght_rooms, unsigned short* ward_indeces, bool *available) {
#if defined(DEBUG) && !defined(ENSEMBLE)
        printf("5,%d,%d,Beginning of findFreeRoomOfTypeAndArea for agent with id %d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID));
#endif
        const int agent_type = FLAMEGPU->getVariable<int>(AGENT_TYPE);
        const short contacts_id = FLAMEGPU->getVariable<short>(CONTACTS_ID);

        int random_iterator = random;
        unsigned short final_target;

        auto spawnrooms_areas_ids = FLAMEGPU->environment.getMacroProperty<unsigned short, NUM_AREAS, NUM_SPAWNROOM + 1>(SPAWNROOMS_AREAS_IDS);
        auto global_resources = FLAMEGPU->environment.getMacroProperty<int, V>(GLOBAL_RESOURCES);
        auto global_resources_counter = FLAMEGPU->environment.getMacroProperty<unsigned int, V>(GLOBAL_RESOURCES_COUNTER);
        auto specific_resources = FLAMEGPU->environment.getMacroProperty<int, NUMBER_OF_AGENTS_TYPES, V>(SPECIFIC_RESOURCES);
        auto specific_resources_counter = FLAMEGPU->environment.getMacroProperty<unsigned int, NUMBER_OF_AGENTS_TYPES, V>(SPECIFIC_RESOURCES_COUNTER);

        unsigned short random_area = (unsigned short) (cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_UNIFORM_0_1_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX)], UNIFORM, contacts_id, 0.0f, (float) (NUM_AREAS-1), false));
        final_target = (unsigned short) spawnrooms_areas_ids[random_area][(unsigned short) (cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_UNIFORM_0_1_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX)], UNIFORM, contacts_id, 1.0f, (float) spawnrooms_areas_ids[random_area][0], false))];

        auto messages = FLAMEGPU->message_in(flow);
        bool room_resources = false;
        do {
            auto list_front = messages.begin();

            for(int i = 0; i < ward_indeces[random_iterator]; i++) list_front++;

            final_target = (*list_front).getVariable<short>(GRAPH_NODE);

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
        printf("5,%d,%d,Ending of findFreeRoomOfTypeAndArea for agent with id %d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID));
#endif
        return final_target;
    }

    /**
     * Take the next destination inside the determined flow of the agent.
    */
    FLAMEGPU_DEVICE_FUNCTION short take_new_destination_flow(DeviceAPI<MessageBucket, MessageBucket>* FLAMEGPU, int *stay, const short start_node, bool *available, const bool identified = false, const unsigned short severity = MINOR){
#if defined(DEBUG) && !defined(ENSEMBLE)
        printf("5,%d,%d,Beginning of take_new_destination_flow for agent with id %d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID));
#endif
        unsigned short flow_index = FLAMEGPU->getVariable<unsigned short>(FLOW_INDEX) + 1;
        unsigned short week_day_flow = FLAMEGPU->getVariable<unsigned short>(WEEK_DAY_FLOW);
        short final_target;

        const short start_node_type = FLAMEGPU->environment.getProperty<short, V>(NODE_TYPE, start_node);
        const unsigned short day = FLAMEGPU->environment.getProperty<unsigned short>(DAY);
        const int agent_type = FLAMEGPU->getVariable<int>(AGENT_TYPE);
        const int agent_subtype = FLAMEGPU->getVariable<int>(AGENT_SUBTYPE);
        const short contacts_id = FLAMEGPU->getVariable<short>(CONTACTS_ID);

        auto spawnrooms_areas_ids = FLAMEGPU->environment.getMacroProperty<unsigned short, NUM_AREAS, NUM_SPAWNROOM + 1>(SPAWNROOMS_AREAS_IDS);
        auto env_flow = FLAMEGPU->environment.getMacroProperty<int, NUMBER_OF_AGENTS_TYPES, NUMBER_OF_AGENTS_SUBTYPES, DAYS_IN_A_WEEK, FLOW_LENGTH>(ENV_FLOW);
        auto env_flow_distr = FLAMEGPU->environment.getMacroProperty<int, NUMBER_OF_AGENTS_TYPES, NUMBER_OF_AGENTS_SUBTYPES, DAYS_IN_A_WEEK, FLOW_LENGTH>(ENV_FLOW_DISTR);
        auto env_flow_distr_firstparam = FLAMEGPU->environment.getMacroProperty<int, NUMBER_OF_AGENTS_TYPES, NUMBER_OF_AGENTS_SUBTYPES, DAYS_IN_A_WEEK, FLOW_LENGTH>(ENV_FLOW_DISTR_FIRSTPARAM);
        auto env_flow_distr_secondparam = FLAMEGPU->environment.getMacroProperty<int, NUMBER_OF_AGENTS_TYPES, NUMBER_OF_AGENTS_SUBTYPES, DAYS_IN_A_WEEK, FLOW_LENGTH>(ENV_FLOW_DISTR_SECONDPARAM);
        auto env_flow_area = FLAMEGPU->environment.getMacroProperty<int, NUMBER_OF_AGENTS_TYPES, NUMBER_OF_AGENTS_SUBTYPES, DAYS_IN_A_WEEK, FLOW_LENGTH>(ENV_FLOW_AREA);
        auto env_room_for_quarantine_type = FLAMEGPU->environment.getMacroProperty<int, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_ROOM_FOR_QUARANTINE_TYPE);
        auto env_room_for_quarantine_area = FLAMEGPU->environment.getMacroProperty<int, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_ROOM_FOR_QUARANTINE_AREA);

        // Resources
        auto global_resources = FLAMEGPU->environment.getMacroProperty<int, V>(GLOBAL_RESOURCES);
        auto global_resources_counter = FLAMEGPU->environment.getMacroProperty<unsigned int, V>(GLOBAL_RESOURCES_COUNTER);
        auto specific_resources = FLAMEGPU->environment.getMacroProperty<int, NUMBER_OF_AGENTS_TYPES, V>(SPECIFIC_RESOURCES);
        auto specific_resources_counter = FLAMEGPU->environment.getMacroProperty<unsigned int, NUMBER_OF_AGENTS_TYPES, V>(SPECIFIC_RESOURCES_COUNTER);
        auto alternative_resources_area_det = FLAMEGPU->environment.getMacroProperty<int, NUMBER_OF_AGENTS_TYPES, V>(ALTERNATIVE_RESOURCES_AREA_DET);
        auto alternative_resources_type_det = FLAMEGPU->environment.getMacroProperty<int, NUMBER_OF_AGENTS_TYPES, V>(ALTERNATIVE_RESOURCES_TYPE_DET);

        const int flow = (int) env_flow[agent_type][agent_subtype][week_day_flow][flow_index];
        const int flow_area = (int) env_flow_area[agent_type][agent_subtype][week_day_flow][flow_index];

        FLAMEGPU->setVariable<unsigned short>(FLOW_INDEX, flow_index);

        // const bool isValidFlow = flow != -1 && flow != SPAWNROOM;
        if(env_flow_distr[agent_type][agent_subtype][week_day_flow][flow_index] != -1)
            *stay = (unsigned int) cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_FLOW_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX)], env_flow_distr[agent_type][agent_subtype][week_day_flow][flow_index], contacts_id, (float) env_flow_distr_firstparam[agent_type][agent_subtype][week_day_flow][flow_index], (float) env_flow_distr_secondparam[agent_type][agent_subtype][week_day_flow][flow_index], true);

        unsigned short j = 0;
        if(flow != SPAWNROOM || (severity == MAJOR && (int) env_room_for_quarantine_type[day-1][agent_type] != SPAWNROOM)){
            auto messages = FLAMEGPU->message_in(flow);
            int area = flow_area;

            if(severity == MAJOR){
                messages = FLAMEGPU->message_in((int) env_room_for_quarantine_type[day-1][agent_type]);
                area = (int) env_room_for_quarantine_area[day-1][agent_type];
            }

            unsigned short ward_indeces[SOLUTION_LENGTH];

            auto i = messages.begin();
            unsigned short k = 0;
            while(i != messages.end()){
                const int local_area = (*i).getVariable<int>(AREA);

                if(local_area == area){
                    ward_indeces[j] = k;
                    j++;
                }

                i++;
                k++;
            }

            const short contacts_id = FLAMEGPU->getVariable<short>(CONTACTS_ID);

            int random = round(cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_TAKE_NEW_DESTINATION_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX)], UNIFORM, contacts_id, 0.0f, (float) (j-1), false));
            int random_iterator = random;
            int lenght_rooms = j;
            unsigned int get_global_resource;
            unsigned int get_specific_resource;

            // If the agent is already waiting for a node, go for it
            if(FLAMEGPU->getVariable<short>(NODE_WAITING_FOR) != -1){
                final_target = FLAMEGPU->getVariable<short>(NODE_WAITING_FOR);
                *available = true;
            }
            else {
                auto list_front = messages.begin();

                for(int i = 0; i < ward_indeces[random]; i++) list_front++;

                final_target = (*list_front).getVariable<short>(GRAPH_NODE);
            }

            if(severity == MINOR){
                if(alternative_resources_type_det[agent_type][final_target] == WAITINGROOM && FLAMEGPU->getVariable<int>(WAITING_ROOM_FLAG) == OUTSIDE_WAITING_ROOM){

                    *available = true;
                    float agent_pos[3] = {FLAMEGPU->getVariable<float>(X), FLAMEGPU->getVariable<float>(Y), FLAMEGPU->getVariable<float>(Z)};
                    FLAMEGPU->setVariable<short>(NODE_WAITING_FOR, final_target);
                    float min_separation = numeric_limits<float>::max();

                    for(const auto& message: FLAMEGPU->message_in(WAITINGROOM)) {
                        const unsigned short near_agent_pos[3] = {message.getVariable<unsigned short>(X), message.getVariable<unsigned short>(Y), message.getVariable<unsigned short>(Z)};

                        float separation = abs(near_agent_pos[0] - agent_pos[0]) + abs(near_agent_pos[1] - agent_pos[1]) + abs(near_agent_pos[2] - agent_pos[2]);
                        if(separation < min_separation){
                            min_separation = separation;
                            final_target = message.getVariable<short>(GRAPH_NODE);
                        }
                    }

                    FLAMEGPU->setVariable<int>(WAITING_ROOM_FLAG, INSIDE_WAITING_ROOM);
                    FLAMEGPU->setVariable<int>(ENTRY_EXIT_FLAG, STAYING_IN_WAITING_ROOM);
                    *stay = 2;

                }
                else if(alternative_resources_type_det[agent_type][final_target] == WAITINGROOM && FLAMEGPU->getVariable<int>(WAITING_ROOM_FLAG) == INSIDE_WAITING_ROOM){

                    //The agent have waited in waiting room and now go to the right flux room
                    FLAMEGPU->setVariable<int>(WAITING_ROOM_FLAG, OUTSIDE_WAITING_ROOM);
                    final_target = FLAMEGPU->getVariable<short>(NODE_WAITING_FOR);
                    FLAMEGPU->setVariable<short>(NODE_WAITING_FOR, -1);
                    *available = true;

                }
                else if(alternative_resources_type_det[agent_type][final_target] != WAITINGROOM){

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

                    //if the initial room is not avaiable because the resources are over, explore the alternatives:
                    if(!*available){
                        get_specific_resource = --specific_resources_counter[agent_type][final_target];

                        //search another room of the same type and area
                        if(alternative_resources_area_det[agent_type][final_target] == area && alternative_resources_type_det[agent_type][final_target] == flow){
                            final_target = findFreeRoomOfTypeAndArea(FLAMEGPU, flow, random, lenght_rooms, ward_indeces, available);
                        }
                        //search another room of the alternative
                        else if(alternative_resources_area_det[agent_type][final_target] != area || alternative_resources_type_det[agent_type][final_target] != flow){

                            auto messages = FLAMEGPU->message_in(alternative_resources_type_det[agent_type][final_target]);

                            unsigned short ward_indeces_alternative[SOLUTION_LENGTH];
                            unsigned short j = 0;
                            unsigned short k = 0;

                            auto i = messages.begin();
                            while(i != messages.end()){
                                const int local_area = (*i).getVariable<int>(AREA);

                                if(local_area == alternative_resources_area_det[agent_type][final_target]){
                                    ward_indeces_alternative[j] = k;
                                    j++;
                                }

                                i++;
                                k++;
                            }
                            int random = round(cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_TAKE_NEW_DESTINATION_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX)], UNIFORM, contacts_id, 0.0f, (float) (j-1), false));
                            final_target = findFreeRoomOfTypeAndArea(FLAMEGPU, alternative_resources_type_det[agent_type][final_target], random, lenght_rooms, ward_indeces_alternative, available);
                        }
                    }
                }

                if(!*available || alternative_resources_type_det[agent_type][final_target] == -1){
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
            unsigned short spawnroom_area = flow_area;
            // if(flow == -1){
            //     spawnroom_area = env_flow_area[agent_type][agent_subtype][week_day_flow][flow_index];
            // }

            final_target = (unsigned short) spawnrooms_areas_ids[spawnroom_area][(unsigned short) (cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_UNIFORM_0_1_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX)], UNIFORM, contacts_id, 1.0f, (float) spawnrooms_areas_ids[spawnroom_area][0], false))];
        }

        FLAMEGPU->setVariable<float, 3>(FINAL_TARGET, 0, FLAMEGPU->environment.getProperty<unsigned short, V>(INDEX2COORDX, final_target));
        FLAMEGPU->setVariable<float, 3>(FINAL_TARGET, 1, FLAMEGPU->environment.getProperty<unsigned short, V>(INDEX2COORDY, final_target));
        FLAMEGPU->setVariable<float, 3>(FINAL_TARGET, 2, FLAMEGPU->environment.getProperty<unsigned short, V>(INDEX2COORDZ, final_target));

        if(identified == IDENTIFIED)
            FLAMEGPU->setVariable<int>(ROOM_FOR_QUARANTINE_INDEX, final_target);

        if((int) env_flow[agent_type][agent_subtype][week_day_flow][flow_index + 1] == -1 && CHECK_IS_SPAWNROOM(final_target))
            *stay = 1;

#if defined(DEBUG) && !defined(ENSEMBLE)
        printf("5,%d,%d,Ending of take_new_destination_flow for agent with id %d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID));
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

        //Initialize the starting node
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
     * Find the shortest path between two cells in matrix
     * TOFINISHNOTALREADYUSED A STAR IN UNA MATRICE
    */
    // FLAMEGPU_DEVICE_FUNCTION void a_star_matrix( DeviceAPI<MessageBucket, MessageBucket>* FLAMEGPU,  const unsigned short start_idx, const unsigned short goal_idx, short* solution) {
    //     // ClosedSet: Mappa [Indice Nodo] -> [Indice Padre]
    //     // Serve sia per sapere se visitato, sia per ricostruire il percorso
    //     short closedset[GRID_SIZE];

    //     // OpenSet: Mappa [Indice Nodo] -> {F_Cost, G_Cost, Parent}
    //     // Usiamo array statici per evitare malloc/new che su GPU non esistono/sono lenti
    //     short openset[GRID_SIZE][3];

    //     // Inizializzazione rapida a "Vuoto"
    //     for (unsigned short i = 0; i < GRID_SIZE; ++i) {
    //         closedset[i] = NOT_PRESENT;
    //         openset[i][F_COST] = NOT_PRESENT;
    //         openset[i][G_COST] = NOT_PRESENT;
    //         openset[i][PARENT] = NOT_PRESENT;
    //     }

    //     // Coordinate Start/Goal
    //     short start_x = start_idx % MAP_DIM_X;
    //     short start_y = start_idx / MAP_DIM_X;
    //     short goal_x = goal_idx % MAP_DIM_X;
    //     short goal_y = goal_idx / MAP_DIM_X;

    //     // Recuperiamo la mappa dall'Environment (Global Read-Only Memory)
    //     // Assumiamo esista una proprietà macro 'grid_map' contenente 1 (walkable) e 0 (wall)
    //     auto grid_data = FLAMEGPU->environment.getMacroProperty<unsigned short, GRID_SIZE>("grid_map");

    //     // Check banale: Se start o goal sono muri, esci subito
    //     if (grid_data[start_idx] == WALL || grid_data[goal_idx] == WALL) {
    //         solution[0] = -1;
    //         return;
    //     }

    //     // Inseriamo il nodo di partenza
    //     short initial_h = CHEBYSHEV_DISTANCE(start_x, goal_x, start_y, goal_y);
    //     openset[start_idx][F_COST] = initial_h; // G=0, quindi F = H
    //     openset[start_idx][G_COST] = 0;
    //     openset[start_idx][PARENT] = STARTING_POINT;

    //     // n_open tiene traccia di quanti nodi ci sono da esplorare
    //     for(unsigned short n_open = 1; n_open > 0;) {

    //         // 1. Trova il nodo con F_COST minore (Lowest F)
    //         // Su GPU non abbiamo priority_queue, facciamo una scansione lineare (O(N))
    //         // Questo è il collo di bottiglia, ma con short array in cache è accettabile.
    //         short current_idx = NOT_PRESENT;
    //         short min_f = 32000; // Valore sentinella (MAX_SHORT approx)

    //         for(int i = 0; i < GRID_SIZE; ++i) {
    //             short f = openset[i][F_COST];
    //             if (f != NOT_PRESENT) {
    //                 if (f < min_f) {
    //                     min_f = f;
    //                     current_idx = i;
    //                 }
    //             }
    //         }

    //         // Safety break
    //         if (current_idx == NOT_PRESENT) break;

    //         // Recuperiamo i dati del nodo corrente prima di chiuderlo
    //         short current_g = openset[current_idx][G_COST];
    //         short current_parent = openset[current_idx][PARENT];

    //         // 2. Sposta da OpenSet a ClosedSet
    //         openset[current_idx][F_COST] = NOT_PRESENT; // Rimuovi da open
    //         openset[current_idx][G_COST] = NOT_PRESENT;
    //         openset[current_idx][PARENT] = NOT_PRESENT;
    //         n_open--;

    //         closedset[current_idx] = current_parent; // Segna come visitato

    //         // 3. Controllo vittoria (Siamo arrivati?)
    //         if (current_idx == goal_idx) {
    //             // BACKTRACKING DEL PERCORSO
    //             short temp_path[MAX_MATRIX_SOLUTION_LENGTH];
    //             short length = 0;
    //             short backtrack = current_idx;

    //             // Risaliamo la catena dei padri
    //             while (backtrack != STARTING_POINT && length < MAX_MATRIX_SOLUTION_LENGTH) {
    //                 temp_path[length++] = backtrack;
    //                 backtrack = closedset[backtrack]; // Salta al padre
    //             }

    //             // Invertiamo l'array per averlo da Start -> Goal
    //             for(unsigned short i = 0, j = length - 1; i < length; ++i, --j){
    //                 solution[i] = temp_path[j];
    //             }
    //             // Riempiamo il resto con -1
    //             for(unsigned short i = length; i < MAX_MATRIX_SOLUTION_LENGTH; ++i) {
    //                 solution[i] = -1;
    //             }
    //             return; // Fine successo
    //         }

    //         // 4. Espansione Vicini (8 Direzioni)
    //         short c_x = current_idx % MAP_DIM_X;
    //         short c_y = current_idx / MAP_DIM_X;

    //         // Look-up table per le 8 direzioni (dx, dy)
    //         const short dirs_x[8] = {0, 0, -1, 1, -1, 1, -1, 1};
    //         const short dirs_y[8] = {-1, 1, 0, 0, -1, -1, 1, 1};

    //         for (int k = 0; k < 8; ++k) {
    //             short nx = c_x + dirs_x[k];
    //             short ny = c_y + dirs_y[k];

    //             if (IS_VALID(nx, ny)) {
    //                 short n_idx = IDX(nx, ny);

    //                 // Se è calpestabile E non è stato già chiuso
    //                 if (grid_data[n_idx] == WALKABLE && closedset[n_idx] == NOT_PRESENT) {

    //                     // Costo movimento: 1 per tutti (Coerente con Chebyshev)
    //                     short new_g = current_g + 1;

    //                     short old_g = openset[n_idx][G_COST];
    //                     bool is_in_open = (old_g != NOT_PRESENT);

    //                     // Se non è in open OPPURE abbiamo trovato una strada più corta
    //                     if (!is_in_open || new_g < old_g) {

    //                         // Calcolo Euristica
    //                         short h = CHEBYSHEV_DISTANCE(nx, goal_x, ny, goal_y);

    //                         openset[n_idx][F_COST] = new_g + h;
    //                         openset[n_idx][G_COST] = new_g;
    //                         openset[n_idx][PARENT] = current_idx;

    //                         if (!is_in_open) {
    //                             n_open++;
    //                         }
    //                     }
    //                 }
    //             }
    //         }
    //     }

    //     // Se arriviamo qui, open set è vuoto e non abbiamo trovato il goal
    //     solution[0] = -1;
    // }

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

            stay_matrix[contacts_id][*target_index].exchange(0);

            if(next_index != *target_index){
                *target_index = (next_index + 1) % SOLUTION_LENGTH;
            }
        }

        short i = 1;
        while(i < SOLUTION_LENGTH && new_targets[i] != -1){
            float jitter_x = 0.0f;
            float jitter_z = 0.0f;

            // Generate a random offset
            if(i+1 < SOLUTION_LENGTH && new_targets[i+1] == -1){
                generate_offset(FLAMEGPU, &jitter_x, &jitter_z, new_targets[i]);
            }

            float x = FLAMEGPU->environment.template getProperty<float, V>(NODE_X, new_targets[i]);
            float z = FLAMEGPU->environment.template getProperty<float, V>(NODE_Z, new_targets[i]);

            new_target_x = x + jitter_x;
            new_target_y = FLAMEGPU->environment.template getProperty<unsigned short, V>(INDEX2COORDY, new_targets[i]);
            new_target_z = z + jitter_z;

            intermediate_target_x[contacts_id][*target_index].exchange(new_target_x);
            intermediate_target_y[contacts_id][*target_index].exchange(new_target_y);
            intermediate_target_z[contacts_id][*target_index].exchange(new_target_z);

            *target_index = (*target_index + 1) % SOLUTION_LENGTH;

            ++i;
        }

        stay_matrix[contacts_id][*target_index].exchange(stay);
        FLAMEGPU->template setVariable<unsigned short>(TARGET_INDEX, *target_index);

        // printf("DEBUG-TARGETS: Agente %d -> Ha assegnato STAY=%d all'indice array [%d]. Coord(X:%f, Z:%f)\n",
        //        contacts_id, stay, *target_index,
        //        (float)intermediate_target_x[contacts_id][*target_index],
        //        (float)intermediate_target_z[contacts_id][*target_index]);

#if defined(DEBUG) && !defined(ENSEMBLE)
        printf("5,%d,%d,Ending update_targets for agent with id %d\n", FLAMEGPU->environment.template getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->template getVariable<short>(CONTACTS_ID));
#endif
    }

    /**
     * Update agent's flow for the next day in which the agent will enter in the environment.
    */
    FLAMEGPU_DEVICE_FUNCTION void update_flow(DeviceAPI<MessageBucket, MessageBucket>* FLAMEGPU, const bool quarantine){
#if defined(DEBUG) && !defined(ENSEMBLE)
        printf("5,%d,%d,Beginning of update_flow for agent with id %d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID));
#endif
        const int agent_type = FLAMEGPU->getVariable<int>(AGENT_TYPE);
        const int agent_subtype = FLAMEGPU->getVariable<int>(AGENT_SUBTYPE);
        const short contacts_id = FLAMEGPU->getVariable<short>(CONTACTS_ID);
        const unsigned short target_index = FLAMEGPU->getVariable<unsigned short>(TARGET_INDEX);
        const unsigned short identified = FLAMEGPU->getVariable<unsigned short>(IDENTIFIED_INFECTED);

        auto stay_matrix = FLAMEGPU->environment.getMacroProperty<unsigned int, TOTAL_AGENTS_ESTIMATION, SOLUTION_LENGTH>(STAY);
        auto env_flow = FLAMEGPU->environment.getMacroProperty<int, NUMBER_OF_AGENTS_TYPES, NUMBER_OF_AGENTS_SUBTYPES, DAYS_IN_A_WEEK, FLOW_LENGTH>(ENV_FLOW);
        auto env_flow_distr = FLAMEGPU->environment.getMacroProperty<int, NUMBER_OF_AGENTS_TYPES, NUMBER_OF_AGENTS_SUBTYPES, DAYS_IN_A_WEEK, FLOW_LENGTH>(ENV_FLOW_DISTR);
        auto env_flow_distr_firstparam = FLAMEGPU->environment.getMacroProperty<int, NUMBER_OF_AGENTS_TYPES, NUMBER_OF_AGENTS_SUBTYPES, DAYS_IN_A_WEEK, FLOW_LENGTH>(ENV_FLOW_DISTR_FIRSTPARAM);
        auto env_flow_distr_secondparam = FLAMEGPU->environment.getMacroProperty<int, NUMBER_OF_AGENTS_TYPES, NUMBER_OF_AGENTS_SUBTYPES, DAYS_IN_A_WEEK, FLOW_LENGTH>(ENV_FLOW_DISTR_SECONDPARAM);
        auto env_hours_schedule = FLAMEGPU->environment.getMacroProperty<int, NUMBER_OF_AGENTS_TYPES, NUMBER_OF_AGENTS_SUBTYPES, DAYS_IN_A_WEEK, HOURS_SCHEDULE_LENGTH>(ENV_HOURS_SCHEDULE);

        unsigned short entry_time_index = FLAMEGPU->getVariable<unsigned short>(ENTRY_TIME_INDEX) + 1;
        unsigned short week_day_flow = FLAMEGPU->getVariable<unsigned short>(WEEK_DAY_FLOW);
        unsigned short empty_days;
        int start_step;

        if((int) env_hours_schedule[agent_type][agent_subtype][week_day_flow][2 * entry_time_index] == 0 || quarantine){
            entry_time_index = 0;
            week_day_flow = (week_day_flow + 1) % DAYS_IN_A_WEEK;

            empty_days = 0;
            while((int) env_flow[agent_type][agent_subtype][week_day_flow][0] == -1){
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
            unsigned int stay_flow_index_0 = (unsigned int) cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_FLOW_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX)], env_flow_distr[agent_type][agent_subtype][week_day_flow][0], contacts_id, (float) env_flow_distr_firstparam[agent_type][agent_subtype][week_day_flow][0], (float) env_flow_distr_secondparam[agent_type][agent_subtype][week_day_flow][0], true);
            stay_matrix[contacts_id][target_index].exchange((unsigned int) (empty_days * STEPS_IN_A_DAY + (STEPS_IN_A_DAY - ((FLAMEGPU->getStepCounter() + START_STEP_TIME) % STEPS_IN_A_DAY)) + start_step + stay_flow_index_0));
            FLAMEGPU->setVariable<unsigned short>(FLOW_INDEX, 0);
        }
        else{
            stay_matrix[contacts_id][target_index].exchange((unsigned int) start_step);
        }

        FLAMEGPU->setVariable<unsigned short>(ENTRY_TIME_INDEX, entry_time_index);
        FLAMEGPU->setVariable<unsigned short>(WEEK_DAY_FLOW, week_day_flow);
#if defined(DEBUG) && !defined(ENSEMBLE)
        printf("5,%d,%d,Ending update_flow for agent with id %d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID));
#endif
    }

    FLAMEGPU_DEVICE_FUNCTION void put_in_quarantine(DeviceAPI<MessageBucket, MessageBucket> *FLAMEGPU){
#if defined(DEBUG) && !defined(ENSEMBLE)
        printf("5,%d,%d,Beginning put_in_quarantine for agent with id %d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID));
#endif
        const unsigned short day = FLAMEGPU->environment.getProperty<unsigned short>(DAY);
        const short contacts_id = FLAMEGPU->getVariable<short>(CONTACTS_ID);
        const int agent_type = FLAMEGPU->getVariable<int>(AGENT_TYPE);

        auto stay_matrix = FLAMEGPU->environment.getMacroProperty<unsigned int, TOTAL_AGENTS_ESTIMATION, SOLUTION_LENGTH>(STAY);
        auto env_quarantine_days_distr = FLAMEGPU->environment.getMacroProperty<int, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_QUARANTINE_DAYS_DISTR);
        auto env_quarantine_days_distr_firstparam = FLAMEGPU->environment.getMacroProperty<int, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_QUARANTINE_DAYS_DISTR_FIRSTPARAM);
        auto env_quarantine_days_distr_secondparam = FLAMEGPU->environment.getMacroProperty<int, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_QUARANTINE_DAYS_DISTR_SECONDPARAM);
        auto env_quarantine_swab_days_distr = FLAMEGPU->environment.getMacroProperty<int, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_QUARANTINE_SWAB_DAYS_DISTR);
        auto env_quarantine_swab_days_distr_firstparam = FLAMEGPU->environment.getMacroProperty<float, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_QUARANTINE_SWAB_DAYS_DISTR_FIRSTPARAM);
        auto env_quarantine_swab_days_distr_secondparam = FLAMEGPU->environment.getMacroProperty<float, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_QUARANTINE_SWAB_DAYS_DISTR_SECONDPARAM);

        unsigned short quarantine = FLAMEGPU->getVariable<unsigned short>(QUARANTINE);
        unsigned short identified_bool = FLAMEGPU->getVariable<unsigned short>(IDENTIFIED_INFECTED);
        unsigned short severity = FLAMEGPU->getVariable<unsigned short>(SEVERITY);
        unsigned short target_index = FLAMEGPU->getVariable<unsigned short>(TARGET_INDEX);
        bool already_in_quarantine = quarantine > 0;

        quarantine = cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_QUARANTINE_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX)], (int) env_quarantine_days_distr[day-1][agent_type], contacts_id, (float) env_quarantine_days_distr_firstparam[day-1][agent_type], (float) env_quarantine_days_distr_secondparam[day-1][agent_type], true);
        FLAMEGPU->setVariable<unsigned short>(QUARANTINE, quarantine);

        int stay = 1;

        auto coord2index = FLAMEGPU->environment.getMacroProperty<short, FLOORS, ENV_DIM_Z, ENV_DIM_X>(COORD2INDEX);

        const float final_target[3] = {FLAMEGPU->getVariable<float, 3>(FINAL_TARGET, 0), FLAMEGPU->getVariable<float, 3>(FINAL_TARGET, 1), FLAMEGPU->getVariable<float, 3>(FINAL_TARGET, 2)};
        const unsigned short agent_with_a_rate = FLAMEGPU->getVariable<unsigned short>(AGENT_WITH_A_RATE);

        stay_matrix[contacts_id][target_index].exchange(0);

        auto global_resources_counter = FLAMEGPU->environment.getMacroProperty<unsigned int, V>(GLOBAL_RESOURCES_COUNTER);
        auto specific_resources_counter = FLAMEGPU->environment.getMacroProperty<unsigned int, NUMBER_OF_AGENTS_TYPES, V>(SPECIFIC_RESOURCES_COUNTER);

        if(!already_in_quarantine){
            const short start_node = coord2index[(unsigned short)(final_target[1]/YOFFSET)][(unsigned short)final_target[2]][(unsigned short)final_target[0]];
            const short start_node_type = FLAMEGPU->environment.getProperty<short, V>(NODE_TYPE, start_node);

            if(!CHECK_IS_SPAWNROOM(start_node) && start_node_type != WAITINGROOM){
                --global_resources_counter[start_node];
                --specific_resources_counter[agent_type][start_node];
            }

            const short quarantine_node = take_new_destination_flow(FLAMEGPU, &stay, start_node, NULL, identified_bool, severity);

            auto counters = FLAMEGPU->environment.getMacroProperty<unsigned int, NUM_COUNTERS>(COUNTERS);
            auto spawnrooms_areas_ids = FLAMEGPU->environment.getMacroProperty<unsigned short, NUM_AREAS, NUM_SPAWNROOM + 1>(SPAWNROOMS_AREAS_IDS);

            if(!CHECK_IS_SPAWNROOM(quarantine_node) && !FLAMEGPU->getVariable<unsigned char>(INIT)){
                FLAMEGPU->setVariable<unsigned char>(INIT, 1);

                unsigned short random_area = (unsigned short) cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_UNIFORM_0_1_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX)], UNIFORM, contacts_id, 0.0f, (float) (NUM_AREAS-1), false);
                unsigned short spawnroom_id = GET_SPAWNROOM_ID_FOR_VECTORS((unsigned short) spawnrooms_areas_ids[random_area][(unsigned short) (cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_UNIFORM_0_1_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX)], UNIFORM, contacts_id, 1.0f, (unsigned short) spawnrooms_areas_ids[random_area][0], false))]);

                float x = cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_OFFSET_X_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX)], UNIFORM, contacts_id, FLAMEGPU->environment.getProperty<float, NUM_SPAWNROOM * 4>(EXTERN_RANGES, spawnroom_id * 2), FLAMEGPU->environment.getProperty<float, NUM_SPAWNROOM * 4>(EXTERN_RANGES, (spawnroom_id * 2) + 1), false);
                float y = FLAMEGPU->environment.getProperty<unsigned short, NUM_SPAWNROOM + 1>(ENTRANCE_Y_COORDS, spawnroom_id);
                float z = cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_OFFSET_Z_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX)], UNIFORM, contacts_id, FLAMEGPU->environment.getProperty<float, NUM_SPAWNROOM * 4>(EXTERN_RANGES, (spawnroom_id * 2) + 2 * NUM_SPAWNROOM), FLAMEGPU->environment.getProperty<float, NUM_SPAWNROOM * 4>(EXTERN_RANGES, (spawnroom_id * 2) + 2 * NUM_SPAWNROOM + 1), false);

                FLAMEGPU->setVariable<float>(X, x);
                FLAMEGPU->setVariable<float>(Y, y);
                FLAMEGPU->setVariable<float>(Z, z);

                FLAMEGPU->setVariable<float, 3>(FINAL_TARGET, 0, x);
                FLAMEGPU->setVariable<float, 3>(FINAL_TARGET, 1, y);
                FLAMEGPU->setVariable<float, 3>(FINAL_TARGET, 2, z);
            }

            short solution_start_quarantine[SOLUTION_LENGTH] = {-1};

            a_star(FLAMEGPU, start_node, quarantine_node, solution_start_quarantine);

            update_targets(FLAMEGPU, solution_start_quarantine, &target_index, false, quarantine * STEPS_IN_A_DAY);

            counters[AGENTS_IN_QUARANTINE]++;
        }
        else
            stay_matrix[contacts_id][target_index].exchange(quarantine * STEPS_IN_A_DAY);

        if(agent_with_a_rate)
            FLAMEGPU->setVariable<unsigned short>(FLOW_INDEX, 0);

        if((int) env_quarantine_swab_days_distr[day-1][agent_type] != NO_SWAB){
            int swab_steps = round(cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_QUARANTINE_SWAB_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX)], (int) env_quarantine_swab_days_distr[day-1][agent_type], contacts_id, (float) (STEPS_IN_A_DAY * env_quarantine_swab_days_distr_firstparam[day-1][agent_type]), (float) (STEPS_IN_A_DAY * env_quarantine_swab_days_distr_secondparam[day-1][agent_type]), true));
            FLAMEGPU->setVariable<int>(SWAB_STEPS, swab_steps);
        }

#if defined(DEBUG) && !defined(ENSEMBLE)
        printf("5,%d,%d,Ending put_in_quarantine for agent with id %d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID));
#endif
    }

    /**
     * Make a swab to the agent and handle quarantine.
    */
    FLAMEGPU_DEVICE_FUNCTION void swab(DeviceAPI<MessageBucket, MessageBucket> *FLAMEGPU){
#if defined(DEBUG) && !defined(ENSEMBLE)
        printf("5,%d,%d,Beginning swab for agent with id %d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID));
#endif
        const unsigned short day = FLAMEGPU->environment.getProperty<unsigned short>(DAY);
        const unsigned int disease_state = FLAMEGPU->getVariable<unsigned int>(DISEASE_STATE);
        const short contacts_id = FLAMEGPU->getVariable<short>(CONTACTS_ID);
        const int agent_type = FLAMEGPU->getVariable<int>(AGENT_TYPE);

        auto stay_matrix = FLAMEGPU->environment.getMacroProperty<unsigned int, TOTAL_AGENTS_ESTIMATION, SOLUTION_LENGTH>(STAY);
        auto env_swab_sensitivity = FLAMEGPU->environment.getMacroProperty<float, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_SWAB_SENSITIVITY);
        auto env_swab_specificity = FLAMEGPU->environment.getMacroProperty<float, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_SWAB_SPECIFICITY);
        auto env_swab_distr = FLAMEGPU->environment.getMacroProperty<int, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_SWAB_DISTR);
        auto env_swab_distr_firstparam = FLAMEGPU->environment.getMacroProperty<float, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_SWAB_DISTR_FIRSTPARAM);
        auto env_swab_distr_secondparam = FLAMEGPU->environment.getMacroProperty<float, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_SWAB_DISTR_SECONDPARAM);
        auto env_quarantine_days_distr = FLAMEGPU->environment.getMacroProperty<int, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_QUARANTINE_DAYS_DISTR);
        auto env_quarantine_swab_sensitivity = FLAMEGPU->environment.getMacroProperty<float, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_QUARANTINE_SWAB_SENSITIVITY);
        auto env_quarantine_swab_specificity = FLAMEGPU->environment.getMacroProperty<float, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_QUARANTINE_SWAB_SPECIFICITY);
        auto counters = FLAMEGPU->environment.getMacroProperty<unsigned int, NUM_COUNTERS>(COUNTERS);

        unsigned short identified_bool = FLAMEGPU->getVariable<unsigned short>(IDENTIFIED_INFECTED);
        unsigned short quarantine = FLAMEGPU->getVariable<unsigned short>(QUARANTINE);
        unsigned short severity = FLAMEGPU->getVariable<unsigned short>(SEVERITY);
        unsigned short target_index = FLAMEGPU->getVariable<unsigned short>(TARGET_INDEX);
        bool already_in_quarantine = quarantine > 0;

        if(disease_state == INFECTED){
            const float sensitivity_swab = already_in_quarantine ? (float) env_quarantine_swab_sensitivity[day-1][agent_type]: (float) env_swab_sensitivity[day-1][agent_type];
            float random_sensitivity = cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_UNIFORM_0_1_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX)], UNIFORM, contacts_id, 0.0f, 1.0f, false);

            if(random_sensitivity < sensitivity_swab){
                // True positive
                if(!already_in_quarantine){
                    const unsigned short risk_class = FLAMEGPU->getVariable<unsigned short>(RISK_CLASS);
                    const float severity_covid = FLAMEGPU->environment.getProperty<float, RISK_CLASSES + 1>(VIRUS_SEVERITY, risk_class);

                    float random_severity = cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_UNIFORM_0_1_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX)], UNIFORM, contacts_id, 0.0f, 1.0f, false);

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
            float random_specificity = cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_UNIFORM_0_1_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX)], UNIFORM, contacts_id, 0.0f, 1.0f, false);

            if(random_specificity >= specificity_swab){
                identified_bool = IDENTIFIED;
            }
            else{
                // True negative if random_specificity less or equal (<=) than specificity_swab
                identified_bool = NOT_IDENTIFIED;
            }
        }

        FLAMEGPU->setVariable<unsigned short>(IDENTIFIED_INFECTED, identified_bool);
        FLAMEGPU->setVariable<unsigned short>(SEVERITY, severity);

        counters[SWABS]++;

        FLAMEGPU->setVariable<int>(SWAB_STEPS, 0);

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
            if((int) env_quarantine_days_distr[day-1][agent_type] != NO_QUARANTINE){
                put_in_quarantine(FLAMEGPU);
            }
        }
        else{
            // Not identified (True or False Negative): the agent exits from quarantine
            if(already_in_quarantine){
                stay_matrix[contacts_id][target_index].exchange(1);
            }
            else{
                if((int) env_swab_distr[day-1][agent_type] != NO_SWAB){
                    int swab_steps = round(cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_SWAB_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX)], (int) env_swab_distr[day-1][agent_type], contacts_id, (float) (STEPS_IN_A_DAY * env_swab_distr_firstparam[day-1][agent_type]), (float) (STEPS_IN_A_DAY * env_swab_distr_secondparam[day-1][agent_type]), true));
                    FLAMEGPU->setVariable<int>(SWAB_STEPS, swab_steps);
                }
            }
        }

#if defined(DEBUG) && !defined(ENSEMBLE)
        printf("5,%d,%d,Ending swab for agent with id %d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID));
#endif
    }

    /**
     * The agent exits from quarantine.
    */
    FLAMEGPU_DEVICE_FUNCTION void exit_from_quarantine(DeviceAPI<MessageBucket, MessageBucket> *FLAMEGPU){
#if defined(DEBUG) && !defined(ENSEMBLE)
        printf("5,%d,%d,Beginning exit_from_quarantine for agent with id %d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID));
#endif
        const unsigned short day = FLAMEGPU->environment.getProperty<unsigned short>(DAY);
        const unsigned short agent_with_a_rate = FLAMEGPU->getVariable<unsigned short>(AGENT_WITH_A_RATE);
        const short contacts_id = FLAMEGPU->getVariable<short>(CONTACTS_ID);
        const int agent_type = FLAMEGPU->getVariable<int>(AGENT_TYPE);
        const int agent_subtype = FLAMEGPU->getVariable<int>(AGENT_SUBTYPE);
        const int quarantine_node = FLAMEGPU->getVariable<int>(ROOM_FOR_QUARANTINE_INDEX);

        short solution_quarantine_extern[SOLUTION_LENGTH] = {-1};
        unsigned short target_index = FLAMEGPU->getVariable<unsigned short>(TARGET_INDEX);

        auto spawnrooms_areas_ids = FLAMEGPU->environment.getMacroProperty<unsigned short, NUM_AREAS, NUM_SPAWNROOM + 1>(SPAWNROOMS_AREAS_IDS);
        auto stay_matrix = FLAMEGPU->environment.getMacroProperty<unsigned int, TOTAL_AGENTS_ESTIMATION, SOLUTION_LENGTH>(STAY);
        auto env_hours_schedule = FLAMEGPU->environment.getMacroProperty<int, NUMBER_OF_AGENTS_TYPES, NUMBER_OF_AGENTS_SUBTYPES, DAYS_IN_A_WEEK, HOURS_SCHEDULE_LENGTH>(ENV_HOURS_SCHEDULE);
        auto env_swab_distr = FLAMEGPU->environment.getMacroProperty<int, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_SWAB_DISTR);
        auto env_swab_distr_firstparam = FLAMEGPU->environment.getMacroProperty<float, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_SWAB_DISTR_FIRSTPARAM);
        auto env_swab_distr_secondparam = FLAMEGPU->environment.getMacroProperty<float, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_SWAB_DISTR_SECONDPARAM);
        auto counters = FLAMEGPU->environment.getMacroProperty<unsigned int, NUM_COUNTERS>(COUNTERS);
        auto coord2index = FLAMEGPU->environment.getMacroProperty<short, FLOORS, ENV_DIM_Z, ENV_DIM_X>(COORD2INDEX);

        FLAMEGPU->setVariable<unsigned short>(SEVERITY, MINOR);
        FLAMEGPU->setVariable<int>(SWAB_STEPS, 0);
        FLAMEGPU->setVariable<int>(ROOM_FOR_QUARANTINE_INDEX, -1);
        FLAMEGPU->setVariable<unsigned short>(QUARANTINE, 0);
        FLAMEGPU->setVariable<unsigned short>(IDENTIFIED_INFECTED, NOT_IDENTIFIED);

        unsigned short random_area = (unsigned short) (cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_UNIFORM_0_1_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX)], UNIFORM, contacts_id, 0.0f, (float) (NUM_AREAS-1), false));
        unsigned short extern_node =  (unsigned short) spawnrooms_areas_ids[random_area][(unsigned short) (cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_UNIFORM_0_1_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX)], UNIFORM, contacts_id, 1.0f, (float) spawnrooms_areas_ids[random_area][0], false))];

        a_star(FLAMEGPU, quarantine_node, extern_node, solution_quarantine_extern);

        if(!agent_with_a_rate){
            if(!CHECK_IS_SPAWNROOM(quarantine_node))
                update_targets(FLAMEGPU, solution_quarantine_extern, &target_index, false, 1);
            update_flow(FLAMEGPU, true);
        }
        else{
            unsigned short week_day = (FLAMEGPU->environment.getProperty<unsigned short>(WEEK_DAY) + 1) % DAYS_IN_A_WEEK;

            unsigned short empty_days = 0;
            unsigned short agent_index = week_day;
            while((int) env_hours_schedule[agent_type][agent_subtype][agent_index][1] - (int) env_hours_schedule[agent_type][agent_subtype][agent_index][0] == 0){
                empty_days++;
                agent_index = (agent_index + 1) % DAYS_IN_A_WEEK;
            }

            const unsigned short start_step = (unsigned short) cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_HOURS_SCHEDULE_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX)], UNIFORM, contacts_id, (float) env_hours_schedule[agent_type][agent_subtype][agent_index][0], (float) env_hours_schedule[agent_type][agent_subtype][agent_index][1], true);

            int stay_spawnroom = (STEPS_IN_A_DAY - ((FLAMEGPU->getStepCounter() + START_STEP_TIME) % STEPS_IN_A_DAY)) + start_step;

            if(!CHECK_IS_SPAWNROOM(quarantine_node))
                update_targets(FLAMEGPU, solution_quarantine_extern, &target_index, false, stay_spawnroom);
            else
                stay_matrix[contacts_id][target_index].exchange(stay_spawnroom);
        }

        if((int) env_swab_distr[day-1][agent_type] != NO_SWAB){
            int swab_steps = round(cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_SWAB_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX)], (int) env_swab_distr[day-1][agent_type], contacts_id, (float) (STEPS_IN_A_DAY * env_swab_distr_firstparam[day-1][agent_type]), (float) (STEPS_IN_A_DAY * env_swab_distr_secondparam[day-1][agent_type]), true));
            FLAMEGPU->setVariable<int>(SWAB_STEPS, swab_steps);
        }

        FLAMEGPU->setVariable<float, 3>(FINAL_TARGET, 0, FLAMEGPU->environment.getProperty<unsigned short, V>(INDEX2COORDX, extern_node));
        FLAMEGPU->setVariable<float, 3>(FINAL_TARGET, 1, FLAMEGPU->environment.getProperty<unsigned short, V>(INDEX2COORDY, extern_node));
        FLAMEGPU->setVariable<float, 3>(FINAL_TARGET, 2, FLAMEGPU->environment.getProperty<unsigned short, V>(INDEX2COORDZ, extern_node));

        float agent_pos[3] = {FLAMEGPU->getVariable<float>(X), FLAMEGPU->getVariable<float>(Y), FLAMEGPU->getVariable<float>(Z)};

        if(agent_pos[1] == INVISIBLE_AGENT_Y)
            printf("0,%d,%d,%d,%d,%f,%f,%f,%d,-1\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID), FLAMEGPU->getVariable<int>(AGENT_TYPE), agent_pos[0], INVISIBLE_AGENT_Y, agent_pos[2], FLAMEGPU->getVariable<int>(DISEASE_STATE));
        else
            printf("0,%d,%d,%d,%d,%f,%f,%f,%d,%d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID), FLAMEGPU->getVariable<int>(AGENT_TYPE), agent_pos[0], agent_pos[1], agent_pos[2], FLAMEGPU->getVariable<int>(DISEASE_STATE), (short) coord2index[(unsigned short)(agent_pos[1]/YOFFSET)][(unsigned short)agent_pos[2]][(unsigned short)agent_pos[0]]);

        if(!CHECK_IS_SPAWNROOM(quarantine_node))
            FLAMEGPU->setVariable<unsigned short>(JUST_EXITED_FROM_QUARANTINE, 1);

        counters[AGENTS_IN_QUARANTINE]--;
#if defined(DEBUG) && !defined(ENSEMBLE)
        printf("5,%d,%d,Ending exit_from_quarantine for agent with id %d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID));
#endif
    }

    /**
     * Handle internal screening.
    */
    FLAMEGPU_DEVICE_FUNCTION void screening(DeviceAPI<MessageBucket, MessageBucket> *FLAMEGPU){
#if defined(DEBUG) && !defined(ENSEMBLE)
        printf("5,%d,%d,Beginning screening for agent with id %d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID));
#endif
        const unsigned short day = FLAMEGPU->environment.getProperty<unsigned short>(DAY);
        const unsigned short identified_bool = FLAMEGPU->getVariable<unsigned short>(IDENTIFIED_INFECTED);
        const int agent_type = FLAMEGPU->getVariable<int>(AGENT_TYPE);

        auto env_swab_distr = FLAMEGPU->environment.getMacroProperty<int, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_SWAB_DISTR);
        auto env_quarantine_swab_days_distr = FLAMEGPU->environment.getMacroProperty<int, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_QUARANTINE_SWAB_DAYS_DISTR);

        int swab_steps = FLAMEGPU->getVariable<int>(SWAB_STEPS);

        swab_steps = swab_steps - 1;
        if(((int) env_swab_distr[day-1][agent_type] != NO_SWAB || (int) env_quarantine_swab_days_distr[day-1][agent_type] != NO_QUARANTINE_SWAB) && swab_steps != -1){
            if(swab_steps){
                FLAMEGPU->setVariable<int>(SWAB_STEPS, swab_steps);
            }
            else{
                swab(FLAMEGPU);
            }
        }
#if defined(DEBUG) && !defined(ENSEMBLE)
        printf("5,%d,%d,Ending screening for agent with id %d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID));
#endif
    }

    /**
     * Handle external screening.
    */
    FLAMEGPU_DEVICE_FUNCTION void external_screening(DeviceAPI<MessageBucket, MessageBucket> *FLAMEGPU){
#if defined(DEBUG) && !defined(ENSEMBLE)
        printf("5,%d,%d,Beginning external_screening for agent with id %d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID));
#endif
        const unsigned short day = FLAMEGPU->environment.getProperty<unsigned short>(DAY);
        const unsigned int disease_state = FLAMEGPU->getVariable<unsigned int>(DISEASE_STATE);
        const short contacts_id = FLAMEGPU->getVariable<short>(CONTACTS_ID);
        const int agent_type = FLAMEGPU->getVariable<int>(AGENT_TYPE);

        auto env_external_screening_first = FLAMEGPU->environment.getMacroProperty<float, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_EXTERNAL_SCREENING_FIRST);
        auto env_external_screening_second = FLAMEGPU->environment.getMacroProperty<float, DAYS, NUMBER_OF_AGENTS_TYPES_PLUS_1>(ENV_EXTERNAL_SCREENING_SECOND);

        unsigned short identified_bool = FLAMEGPU->getVariable<unsigned short>(IDENTIFIED_INFECTED);

        if(identified_bool != IDENTIFIED && FLAMEGPU->getVariable<unsigned short>(EXITED_FROM_ENVIRONMENT)){
            float random_external_screening_first = cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_UNIFORM_0_1_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX)], UNIFORM, contacts_id, 0.0f, 1.0f, false);
            if(random_external_screening_first < (float) env_external_screening_first[day-1][agent_type]){
                swab(FLAMEGPU);
#if defined(DEBUG) && !defined(ENSEMBLE)
                printf("5,%d,%d,Ending external_screening for agent with id %d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID));
#endif
                return;
            }

            float random_external_screening_second = cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_UNIFORM_0_1_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX)], UNIFORM, contacts_id, 0.0f, 1.0f, false);
            if(disease_state == INFECTED && random_external_screening_second < (float) env_external_screening_second[day-1][agent_type]){
                swab(FLAMEGPU);
            }
        }
#if defined(DEBUG) && !defined(ENSEMBLE)
        printf("5,%d,%d,Ending external_screening for agent with id %d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID));
#endif
    }

    /**
     * Handle contagion processes.
    */
    FLAMEGPU_DEVICE_FUNCTION flamegpu::AGENT_STATUS contagion_processes(DeviceAPI<MessageBucket, MessageBucket>* FLAMEGPU){
#if defined(DEBUG) && !defined(ENSEMBLE)
        printf("5,%d,%d,Beginning of contagion_processes for agent with id %d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID));
#endif
        const short contacts_id = FLAMEGPU->getVariable<short>(CONTACTS_ID);
        const unsigned short risk_class = FLAMEGPU->getVariable<unsigned short>(RISK_CLASS);

        int disease_state = FLAMEGPU->getVariable<int>(DISEASE_STATE);
        unsigned short incubation_days = FLAMEGPU->getVariable<unsigned short>(INCUBATION_DAYS);
        unsigned short infection_days = FLAMEGPU->getVariable<unsigned short>(INFECTION_DAYS);
        unsigned short fatality_days = FLAMEGPU->getVariable<unsigned short>(FATALITY_DAYS);;
        unsigned short end_of_immunization_days = FLAMEGPU->getVariable<unsigned short>(END_OF_IMMUNIZATION_DAYS);
        unsigned char infected_contact = FLAMEGPU->getVariable<unsigned char>(INFECTED_CONTACT);

        auto coord2index = FLAMEGPU->environment.getMacroProperty<short, FLOORS, ENV_DIM_Z, ENV_DIM_X>(COORD2INDEX);

        if(disease_state == SUSCEPTIBLE){
            // Contagion through contact
            float contamination_risk = FLAMEGPU->environment.getProperty<float, RISK_CLASSES + 1>(CONTAMINATION_RISK, risk_class);

            const float contamination_risk_decreased_with_mask = FLAMEGPU->environment.getProperty<float>(CONTAMINATION_RISK_DECREASED_WITH_MASK);
            const float virus_variant_factor = FLAMEGPU->environment.getProperty<float>(VIRUS_VARIANT_FACTOR);
            const float area_around_agent = M_PI * (DIAMETER / 2) * (DIAMETER / 2);

            const int mask_type = FLAMEGPU->getVariable<int>(MASK_TYPE);

            contamination_risk = (mask_type != NO_MASK) ? contamination_risk * (1 - contamination_risk_decreased_with_mask): contamination_risk;

            // 0.01 because for now we are not cconsidering symptomatic infected agents (otherwise
            // it would be 1/2^t where t is the time since the beginning of the symptoms)
            const float p_contact = infected_contact > 0 ? ((contamination_risk * 0.01) / area_around_agent): 0.0f;

            // Contagion through aerosol
            const float risk_const = FLAMEGPU->environment.getProperty<float, RISK_CLASSES + 1>(RISK_CONST, risk_class);
            const float quanta_inhaled = FLAMEGPU->getVariable<float>(QUANTA_INHALED);
            const float p_aerosol = 1 - exp(-(quanta_inhaled / risk_const));

            const float p = p_contact + p_aerosol - (p_contact * p_aerosol);

            const float random = cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_UNIFORM_0_1_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX)], UNIFORM, contacts_id, 0.0f, 1.0f, false);

            // See if the agent get the virus
            if(random < p){
#ifdef INCUBATION
                disease_state = EXPOSED;

                incubation_days = (unsigned short) round(cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_INCUBATION_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX)], FLAMEGPU->environment.getProperty<unsigned short, RISK_CLASSES + 1>(MEAN_INCUBATION_DAYS_DISTR, risk_class), contacts_id, (float) FLAMEGPU->environment.getProperty<unsigned short, RISK_CLASSES * 2 + 1>(MEAN_INCUBATION_DAYS, risk_class * 2), (float) FLAMEGPU->environment.getProperty<unsigned short, RISK_CLASSES * 2 + 1>(MEAN_INCUBATION_DAYS, risk_class * 2 + 1), true));
#else
                disease_state = INFECTED;
                infection_days = (unsigned short) round(cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_INFECTION_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX)], FLAMEGPU->environment.getProperty<unsigned short, RISK_CLASSES + 1>(MEAN_INFECTION_DAYS_DISTR, risk_class), contacts_id, (float) FLAMEGPU->environment.getProperty<unsigned short, RISK_CLASSES * 2 + 1>(MEAN_INFECTION_DAYS, risk_class * 2), (float) FLAMEGPU->environment.getProperty<unsigned short, RISK_CLASSES * 2 + 1>(MEAN_INFECTION_DAYS, risk_class * 2 + 1), true));
#ifdef FATALITY
                fatality_days = (unsigned short) round(cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_FATALITY_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX)], FLAMEGPU->environment.getProperty<unsigned short, RISK_CLASSES + 1>(MEAN_FATALITY_DAYS_DISTR, risk_class), contacts_id, (float) FLAMEGPU->environment.getProperty<unsigned short, RISK_CLASSES * 2 + 1>(MEAN_FATALITY_DAYS, risk_class * 2), (float) FLAMEGPU->environment.getProperty<unsigned short, RISK_CLASSES * 2 + 1>(MEAN_FATALITY_DAYS, risk_class * 2 + 1), true));
#endif
#endif
                float agent_pos[3] = {FLAMEGPU->getVariable<float>(X), FLAMEGPU->getVariable<float>(Y), FLAMEGPU->getVariable<float>(Z)};

                if(agent_pos[1] == INVISIBLE_AGENT_Y)
                    printf("0,%d,%d,%d,%d,%f,%f,%f,%d,-1\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID), FLAMEGPU->getVariable<int>(AGENT_TYPE), agent_pos[0], INVISIBLE_AGENT_Y, agent_pos[2], FLAMEGPU->getVariable<int>(DISEASE_STATE));
                else
                    printf("0,%d,%d,%d,%d,%f,%f,%f,%d,%d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID), FLAMEGPU->getVariable<int>(AGENT_TYPE), agent_pos[0], agent_pos[1], agent_pos[2], FLAMEGPU->getVariable<int>(DISEASE_STATE), (short) coord2index[(unsigned short)(agent_pos[1]/YOFFSET)][(unsigned short)agent_pos[2]][(unsigned short)agent_pos[0]]);
            }
        }

        FLAMEGPU->setVariable<float>(QUANTA_INHALED, 0.0f);
        FLAMEGPU->setVariable<unsigned char>(INFECTED_CONTACT, 0);

        FLAMEGPU->setVariable<int>(DISEASE_STATE, disease_state);
        FLAMEGPU->setVariable<unsigned short>(INCUBATION_DAYS, incubation_days);
        FLAMEGPU->setVariable<unsigned short>(INFECTION_DAYS, infection_days);
        FLAMEGPU->setVariable<unsigned short>(FATALITY_DAYS, fatality_days);
        FLAMEGPU->setVariable<unsigned short>(END_OF_IMMUNIZATION_DAYS, end_of_immunization_days);

#if defined(DEBUG) && !defined(ENSEMBLE)
        printf("5,%d,%d,Ending contagion_processes for agent with id %d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID));
#endif
        return ALIVE;
    }

/**
     * Update disease state.
    */
    FLAMEGPU_DEVICE_FUNCTION flamegpu::AGENT_STATUS update_infection(DeviceAPI<MessageBucket, MessageBucket>* FLAMEGPU){
#if defined(DEBUG) && !defined(ENSEMBLE)
        printf("5,%d,%d,Beginning of update_infected for agent with id %d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID));
#endif
        const short contacts_id = FLAMEGPU->getVariable<short>(CONTACTS_ID);
        const unsigned short risk_class = FLAMEGPU->getVariable<unsigned short>(RISK_CLASS);

        int disease_state = FLAMEGPU->getVariable<int>(DISEASE_STATE);
        int disease_state_old = disease_state;
        unsigned short incubation_days = FLAMEGPU->getVariable<unsigned short>(INCUBATION_DAYS);
        unsigned short infection_days = FLAMEGPU->getVariable<unsigned short>(INFECTION_DAYS);
        unsigned short fatality_days = FLAMEGPU->getVariable<unsigned short>(FATALITY_DAYS);
        unsigned short end_of_immunization_days = FLAMEGPU->getVariable<unsigned short>(END_OF_IMMUNIZATION_DAYS);

        auto coord2index = FLAMEGPU->environment.getMacroProperty<short, FLOORS, ENV_DIM_Z, ENV_DIM_X>(COORD2INDEX);

#ifdef INCUBATION
        if(disease_state == EXPOSED){
            if(!incubation_days){
                disease_state = INFECTED;

                infection_days = (unsigned short) round(cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_INFECTION_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX)], FLAMEGPU->environment.getProperty<unsigned short, RISK_CLASSES + 1>(MEAN_INFECTION_DAYS_DISTR, risk_class), contacts_id, (float) FLAMEGPU->environment.getProperty<unsigned short, RISK_CLASSES * 2 + 1>(MEAN_INFECTION_DAYS, risk_class * 2), (float) FLAMEGPU->environment.getProperty<unsigned short, RISK_CLASSES * 2 + 1>(MEAN_INFECTION_DAYS, risk_class * 2 + 1), true));
#ifdef FATALITY
                fatality_days =  (unsigned short) round(cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_FATALITY_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX)], FLAMEGPU->environment.getProperty<unsigned short, RISK_CLASSES + 1>(MEAN_FATALITY_DAYS_DISTR, risk_class), contacts_id, (float) FLAMEGPU->environment.getProperty<unsigned short, RISK_CLASSES * 2 + 1>(MEAN_FATALITY_DAYS, risk_class * 2), (float) FLAMEGPU->environment.getProperty<unsigned short, RISK_CLASSES * 2 + 1>(MEAN_FATALITY_DAYS, risk_class * 2 + 1), true));
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
                printf("5,%d,%d,Ending update_infected for agent with id %d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID));
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
                end_of_immunization_days = (unsigned short) round(cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_END_OF_IMMUNIZATION_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX)], FLAMEGPU->environment.getProperty<unsigned short, RISK_CLASSES + 1>(MEAN_END_OF_IMMUNIZATION_DAYS_DISTR, risk_class), contacts_id, (float) FLAMEGPU->environment.getProperty<unsigned short, RISK_CLASSES * 2 + 1>(MEAN_END_OF_IMMUNIZATION_DAYS, risk_class * 2), (float) FLAMEGPU->environment.getProperty<unsigned short, RISK_CLASSES * 2 + 1>(MEAN_END_OF_IMMUNIZATION_DAYS, risk_class * 2 + 1), true));
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

        FLAMEGPU->setVariable<int>(DISEASE_STATE, disease_state);
        FLAMEGPU->setVariable<unsigned short>(INCUBATION_DAYS, incubation_days);
        FLAMEGPU->setVariable<unsigned short>(INFECTION_DAYS, infection_days);
        FLAMEGPU->setVariable<unsigned short>(FATALITY_DAYS, fatality_days);
        FLAMEGPU->setVariable<unsigned short>(END_OF_IMMUNIZATION_DAYS, end_of_immunization_days);

        if(disease_state != disease_state_old){
            float agent_pos[3] = {FLAMEGPU->getVariable<float>(X), FLAMEGPU->getVariable<float>(Y), FLAMEGPU->getVariable<float>(Z)};

            if(agent_pos[1] == INVISIBLE_AGENT_Y)
                printf("0,%d,%d,%d,%d,%f,%f,%f,%d,-1\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID), FLAMEGPU->getVariable<int>(AGENT_TYPE), agent_pos[0], INVISIBLE_AGENT_Y, agent_pos[2], FLAMEGPU->getVariable<int>(DISEASE_STATE));
            else
                printf("0,%d,%d,%d,%d,%f,%f,%f,%d,%d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID), FLAMEGPU->getVariable<int>(AGENT_TYPE), agent_pos[0], agent_pos[1], agent_pos[2], FLAMEGPU->getVariable<int>(DISEASE_STATE), (short) coord2index[(unsigned short)(agent_pos[1]/YOFFSET)][(unsigned short)agent_pos[2]][(unsigned short)agent_pos[0]]);
        }
#if defined(DEBUG) && !defined(ENSEMBLE)
        printf("5,%d,%d,Ending update_infected for agent with id %d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID));
#endif
        return ALIVE;
    }

    /**
     * Handle outside contagion.
    */
    FLAMEGPU_DEVICE_FUNCTION void outside_contagion(DeviceAPI<MessageBucket, MessageBucket>* FLAMEGPU){
#if defined(DEBUG) && !defined(ENSEMBLE)
        printf("5,%d,%d,Beginning of outside_contagion for agent with id %d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID));
#endif
        if(FLAMEGPU->getVariable<int>(DISEASE_STATE) == SUSCEPTIBLE && FLAMEGPU->getVariable<unsigned short>(EXITED_FROM_ENVIRONMENT) && FLAMEGPU->getVariable<float>(Y) == INVISIBLE_AGENT_Y){
            const short contacts_id = FLAMEGPU->getVariable<short>(CONTACTS_ID);
            const unsigned short day = FLAMEGPU->environment.getProperty<unsigned short>(DAY)-1;
            const float perc_inf = FLAMEGPU->environment.getProperty<float, DAYS + 1>(PERC_INF, day);
            const unsigned short risk_class = FLAMEGPU->getVariable<unsigned short>(RISK_CLASS);

            auto coord2index = FLAMEGPU->environment.getMacroProperty<short, FLOORS, ENV_DIM_Z, ENV_DIM_X>(COORD2INDEX);

            float random = cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_UNIFORM_0_1_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX)], UNIFORM, contacts_id, 0.0f, 1.0f, false);
            if(random < perc_inf){
                auto counters = FLAMEGPU->environment.getMacroProperty<unsigned int, NUM_COUNTERS>(COUNTERS);

                counters[NUM_INFECTED_OUTSIDE]++;

                if(FLAMEGPU->getVariable<unsigned short>(AGENT_WITH_A_RATE) == AGENT_WITHOUT_RATE){

#ifdef INCUBATION
                FLAMEGPU->setVariable<int>(DISEASE_STATE, EXPOSED);

                unsigned short incubation_days = (unsigned short) round(cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_INCUBATION_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX)], FLAMEGPU->environment.getProperty<unsigned short, RISK_CLASSES + 1>(MEAN_INCUBATION_DAYS_DISTR, risk_class), contacts_id, (float) FLAMEGPU->environment.getProperty<unsigned short, RISK_CLASSES * 2 + 1>(MEAN_INCUBATION_DAYS, risk_class * 2), (float) FLAMEGPU->environment.getProperty<unsigned short, RISK_CLASSES * 2 + 1>(MEAN_INCUBATION_DAYS, risk_class * 2 + 1), true));

                FLAMEGPU->setVariable<unsigned short>(INCUBATION_DAYS, incubation_days);
#else
                FLAMEGPU->setVariable<int>(DISEASE_STATE, INFECTED);

                unsigned short infection_days = (unsigned short) round(cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_INFECTION_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX)], FLAMEGPU->environment.getProperty<unsigned short, RISK_CLASSES + 1>(MEAN_INFECTION_DAYS_DISTR, risk_class), contacts_id, (float) FLAMEGPU->environment.getProperty<unsigned short, RISK_CLASSES * 2 + 1>(MEAN_INFECTION_DAYS, risk_class * 2), (float) FLAMEGPU->environment.getProperty<unsigned short, RISK_CLASSES * 2>(MEAN_INFECTION_DAYS, risk_class * 2 + 1), true));
#endif

                }
                else{
                FLAMEGPU->setVariable<int>(DISEASE_STATE, INFECTED);

                unsigned short infection_days = (unsigned short) round(cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_INFECTION_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX)], FLAMEGPU->environment.getProperty<unsigned short, RISK_CLASSES + 1>(MEAN_INFECTION_DAYS_DISTR, risk_class), contacts_id, (float) FLAMEGPU->environment.getProperty<unsigned short, RISK_CLASSES * 2 + 1>(MEAN_INFECTION_DAYS, risk_class * 2), (float) FLAMEGPU->environment.getProperty<unsigned short, RISK_CLASSES * 2>(MEAN_INFECTION_DAYS, risk_class * 2 + 1), true));
            }

#ifdef FATALITY
                unsigned short fatality_days = (unsigned short) round(cuda_pedestrian_rng(FLAMEGPU, PEDESTRIAN_FATALITY_DISTR_IDX, cuda_pedestrian_states[FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX)], FLAMEGPU->environment.getProperty<unsigned short, RISK_CLASSES + 1>(MEAN_FATALITY_DAYS_DISTR, risk_class), contacts_id, (float) FLAMEGPU->environment.getProperty<unsigned short, RISK_CLASSES * 2 + 1>(MEAN_FATALITY_DAYS, risk_class * 2), (float) FLAMEGPU->environment.getProperty<unsigned short, RISK_CLASSES * 2 + 1>(MEAN_FATALITY_DAYS, risk_class * 2 + 1), true));
                FLAMEGPU->setVariable<unsigned short>(FATALITY_DAYS, fatality_days);
#endif
                float agent_pos[3] = {FLAMEGPU->getVariable<float>(X), FLAMEGPU->getVariable<float>(Y), FLAMEGPU->getVariable<float>(Z)};

                if(agent_pos[1] == INVISIBLE_AGENT_Y)
                    printf("0,%d,%d,%d,%d,%f,%f,%f,%d,-1\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID), FLAMEGPU->getVariable<int>(AGENT_TYPE), agent_pos[0], INVISIBLE_AGENT_Y, agent_pos[2], FLAMEGPU->getVariable<int>(DISEASE_STATE));
                else
                    printf("0,%d,%d,%d,%d,%f,%f,%f,%d,%d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID), FLAMEGPU->getVariable<int>(AGENT_TYPE), agent_pos[0], agent_pos[1], agent_pos[2], FLAMEGPU->getVariable<int>(DISEASE_STATE), (short) coord2index[(unsigned short)(agent_pos[1]/YOFFSET)][(unsigned short)agent_pos[2]][(unsigned short)agent_pos[0]]);
            }
        }
#if defined(DEBUG) && !defined(ENSEMBLE)
        printf("5,%d,%d,Ending outside_contagion for agent with id %d\n", FLAMEGPU->environment.getProperty<unsigned short>(RUN_IDX), FLAMEGPU->getStepCounter(), FLAMEGPU->getVariable<short>(CONTACTS_ID));
#endif
    }

    FLAMEGPU_DEVICE_FUNCTION unsigned char findLeftmostIndex(const float target, const float *env_events_cdf, const short num_events) {
        int left = 0;
        int right = num_events - 1;

        if (target > env_events_cdf[1])
            return left;

        if (target <= env_events_cdf[right])
            return right;

        while (left <= right) {
            int mid = left + (right - left) / 2;

            float upper = env_events_cdf[mid];
            float lower = env_events_cdf[mid + 1];

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
