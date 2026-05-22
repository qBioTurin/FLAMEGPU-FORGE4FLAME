#ifndef _MOVEMENT_SUBMODULE_CUH_
#define _MOVEMENT_SUBMODULE_CUH_

#include <vector>
#include <algorithm>
#include "defines.h"

using namespace flamegpu;
using namespace std;



//AGENT FUNCTIONS

FLAMEGPU_AGENT_FUNCTION(avoid_pedestrians, MessageSpatial3D, MessageNone) {
    if (!FLAMEGPU->getVariable<int>(CAN_MOVE)) {
        return ALIVE;
    } 

    const float PI = 3.14159265f;
    const float RAD_PERCEPTION= 45.0f * (PI / 180.0f); 
    
    const float agent_pos[3] = {FLAMEGPU->getVariable<float>(X), FLAMEGPU->getVariable<float>(Y), FLAMEGPU->getVariable<float>(Z)};
    float agent_vel[3] = {FLAMEGPU->getVariable<float>(VELX), FLAMEGPU->getVariable<float>(VELY), FLAMEGPU->getVariable<float>(VELZ)};

    float navigate_velocity[3] = {0.0f, 0.0f, 0.0f};
    float avoid_velocity[3] = {0.0f, 0.0f, 0.0f};

    float speed_sq = agent_vel[0]*agent_vel[0] + agent_vel[2]*agent_vel[2];
    if(speed_sq > 0.0001f){
        float s = sqrtf(speed_sq);
        agent_vel[0] /= s;
        agent_vel[2] /= s;
    }

    for (const auto& message: FLAMEGPU->message_in(agent_pos[0], agent_pos[1], agent_pos[2])){
        const float message_pos[3] = {message.getVariable<float>(X), message.getVariable<float>(Y), message.getVariable<float>(Z)};
        float diff[3] = {
            agent_pos[0] - message_pos[0],
            agent_pos[1] - message_pos[1],
            agent_pos[2] - message_pos[2]
        };

        if (fabsf(diff[1]) > 2.0f) continue;

        float separation = sqrtf(diff[0] * diff[0] + diff[2] * diff[2]);

        if ((separation < FLAMEGPU->message_in.radius()) && (separation > MIN_DISTANCE_STEER)) {
            float to_agent_x = diff[0] / separation;
            float to_agent_z = diff[2] / separation;

            float dot_product = agent_vel[0] * to_agent_x + agent_vel[2] * to_agent_z;
  
            if (dot_product < -1.0f) dot_product = -1.0f;
            if (dot_product > 1.0f) dot_product = 1.0f;
            float ang = acosf(dot_product);
            
            // STEER
            if ( ang < RAD_PERCEPTION || ang > (PI - RAD_PERCEPTION)) {
                float steer_scalar = powf(I_SCALER / separation, 1.25f) * STEER_WEIGHT;
                navigate_velocity[0] += to_agent_x * steer_scalar;
                navigate_velocity[1] += 0.0f;  // No vertical steer
                navigate_velocity[2] += to_agent_z * steer_scalar;
            }

            // AVOID
            float avoid_scalar = powf(I_SCALER / separation, 2.0f) * AVOID_WEIGHT;
            avoid_velocity[0] += to_agent_x * avoid_scalar;
            avoid_velocity[1] += 0.0f; 
            avoid_velocity[2] += to_agent_z * avoid_scalar;
        }
    }

    // maximum velocity rule
    float steer_velocity[3] = {
        navigate_velocity[0] + avoid_velocity[0],
        0.0f,  // No vertical velocity
        navigate_velocity[2] + avoid_velocity[2]
    };

    // Set variables
    FLAMEGPU->setVariable<float>(STEER_X, steer_velocity[0]);
    FLAMEGPU->setVariable<float>(STEER_Y, steer_velocity[1]);
    FLAMEGPU->setVariable<float>(STEER_Z, steer_velocity[2]);

    return ALIVE;
}

FLAMEGPU_AGENT_FUNCTION(outputPedestrianLocationSub, MessageNone, MessageSpatial3D) {
     if (!FLAMEGPU->getVariable<int>(CAN_MOVE)) {
        FLAMEGPU->setVariable<float>(VELX, 0.0f);
        FLAMEGPU->setVariable<float>(VELY, 0.0f);
        FLAMEGPU->setVariable<float>(VELZ, 0.0f); 
    } 
    
    FLAMEGPU->message_out.setVariable<id_t>(ID, FLAMEGPU->getID());
    FLAMEGPU->message_out.setVariable<short>(CONTACTS_ID, FLAMEGPU->getVariable<short>(CONTACTS_ID));
    FLAMEGPU->message_out.setLocation(
        FLAMEGPU->getVariable<float>(X),
        FLAMEGPU->getVariable<float>(Y),
        FLAMEGPU->getVariable<float>(Z)
    );

    return ALIVE;
}

FLAMEGPU_AGENT_FUNCTION(move_agent_function, MessageNone, MessageNone) {
    if (!FLAMEGPU->getVariable<int>(CAN_MOVE)) {
        return ALIVE;
    }

    // Move the agent
    const short contacts_id = FLAMEGPU->getVariable<short>(CONTACTS_ID);
    unsigned short target_index = FLAMEGPU->getVariable<unsigned short>(TARGET_INDEX);
    unsigned short next_index = FLAMEGPU->getVariable<unsigned short>(NEXT_INDEX);
    auto stay_matrix = FLAMEGPU->environment.getMacroProperty<unsigned int, TOTAL_AGENTS_ESTIMATION, SOLUTION_LENGTH>(STAY);
    unsigned int current_stay = (unsigned int) stay_matrix[contacts_id][next_index];
    
    if (current_stay > 0) {
        FLAMEGPU->setVariable<float>(VELX, 0.0f);
        FLAMEGPU->setVariable<float>(VELY, 0.0f);
        FLAMEGPU->setVariable<float>(VELZ, 0.0f);

        return ALIVE;
    }

    float agent_pos[3] = {FLAMEGPU->getVariable<float>(X), FLAMEGPU->getVariable<float>(Y), FLAMEGPU->getVariable<float>(Z)};
    float agent_vel[3] = {FLAMEGPU->getVariable<float>(VELX), FLAMEGPU->getVariable<float>(VELY), FLAMEGPU->getVariable<float>(VELZ)};
    float agent_steer[3] = {FLAMEGPU->getVariable<float>(STEER_X), FLAMEGPU->getVariable<float>(STEER_Y), FLAMEGPU->getVariable<float>(STEER_Z)};
    auto intermediate_target_x = FLAMEGPU->environment.getMacroProperty<float, TOTAL_AGENTS_ESTIMATION, SOLUTION_LENGTH>(INTERMEDIATE_TARGET_X);
    auto intermediate_target_y = FLAMEGPU->environment.getMacroProperty<float, TOTAL_AGENTS_ESTIMATION, SOLUTION_LENGTH>(INTERMEDIATE_TARGET_Y);
    auto intermediate_target_z = FLAMEGPU->environment.getMacroProperty<float, TOTAL_AGENTS_ESTIMATION, SOLUTION_LENGTH>(INTERMEDIATE_TARGET_Z);
    auto coord2index = FLAMEGPU->environment.getMacroProperty<short, FLOORS, ENV_DIM_Z, ENV_DIM_X>(COORD2INDEX);
    float intermediate_target[3] = {(float) intermediate_target_x[contacts_id][next_index], (float) intermediate_target_y[contacts_id][next_index], (float) intermediate_target_z[contacts_id][next_index]};
    unsigned int stay = (unsigned int) stay_matrix[contacts_id][next_index];

    float available_vel = 1.0f;   
    float distance = sqrt(pow(intermediate_target[0] - agent_pos[0], 2) + pow(intermediate_target[1] - agent_pos[1], 2) + pow(intermediate_target[2] - agent_pos[2], 2));
    float arrival_tolerance = (next_index == target_index) ? 1.0f : 0.2f;

    while(distance < available_vel && available_vel > 0.0f){
        if (next_index == target_index && distance <= arrival_tolerance) {
            unsigned int current_stay = (unsigned int) stay_matrix[contacts_id][next_index];
            if (current_stay > 0) {
                available_vel = 0.0f;
                break; 
            }
        }

        agent_pos[0] = intermediate_target[0];
        agent_pos[1] = intermediate_target[1];
        agent_pos[2] = intermediate_target[2];
        available_vel = available_vel - distance;

        next_index = (next_index + 1) % SOLUTION_LENGTH;
        FLAMEGPU->setVariable<unsigned short>(NEXT_INDEX, next_index);
        stay = (unsigned int) stay_matrix[contacts_id][next_index];

        if(next_index != target_index && !stay){
            intermediate_target[0] = (float) intermediate_target_x[contacts_id][next_index];
            intermediate_target[1] = (float) intermediate_target_y[contacts_id][next_index];
            intermediate_target[2] = (float) intermediate_target_z[contacts_id][next_index];
            distance = sqrt(pow(intermediate_target[0] - agent_pos[0], 2) + pow(intermediate_target[1] - agent_pos[1], 2) + pow(intermediate_target[2] - agent_pos[2], 2));
        }
        else {
            arrival_tolerance = 1.0f;
        }

        if (next_index == target_index && distance <= arrival_tolerance) {
            available_vel = 0.0f; 
        }
    }
    
    // Update velocity
    agent_vel[0] = (available_vel * (intermediate_target[0] - agent_pos[0]))/std::max(1.0f, distance);
    agent_vel[1] = (available_vel * (intermediate_target[1] - agent_pos[1]))/std::max(1.0f, distance);
    agent_vel[2] = (available_vel * (intermediate_target[2] - agent_pos[2]))/std::max(1.0f, distance);

    
    //const float MAX_DODGE = 0.15f; to see if substitute in the future
    const float MAX_DODGE = 1.0f / STEP; 
    const float MAX_DODGE_SQ = MAX_DODGE * MAX_DODGE;

    if (available_vel == 0.0f) {
        // Agent has arrived. Turn off avoidance so they don't slide through walls!
        agent_steer[0] = 0.0f;
        agent_steer[2] = 0.0f;
    } else {
        float steer_len_sq = agent_steer[0]*agent_steer[0] + agent_steer[2]*agent_steer[2];
        if (steer_len_sq > MAX_DODGE_SQ) {
            float len = sqrtf(steer_len_sq);
            agent_steer[0] = (agent_steer[0] / len) * MAX_DODGE;
            agent_steer[2] = (agent_steer[2] / len) * MAX_DODGE;
        }
    }

    //Eventually logic to avoid wall? In the future

    float proposed_position[3] = {
        agent_pos[0] + agent_vel[0] + agent_steer[0], 
        agent_pos[1] + agent_vel[1], 
        agent_pos[2] + agent_vel[2] + agent_steer[2]
    };

    agent_pos[0] = proposed_position[0];
    agent_pos[1] = proposed_position[1];
    agent_pos[2] = proposed_position[2];

    agent_vel[0] += agent_steer[0];
    agent_vel[2] += agent_steer[2];

    //graphical trick
    float final_speed = sqrtf(agent_vel[0]*agent_vel[0] + agent_vel[2]*agent_vel[2]);
    if (final_speed > 0.051f) { 
        agent_vel[0] /= final_speed; // Keep heading for view cone
        agent_vel[1] /= final_speed;
        agent_vel[2] /= final_speed;
    } else {
        // The agent has arrived and is just micro-shuffling due to crowd avoidance.
        // FORCE velocity to 0 so they stop drifting and the legs stop animating!
        agent_vel[0] = 0.0f; 
        agent_vel[1] = 0.0f;
        agent_vel[2] = 0.0f;
    }


    // Update variables
    FLAMEGPU->setVariable<float>(X, agent_pos[0]);
    FLAMEGPU->setVariable<float>(Y, agent_pos[1]);
    FLAMEGPU->setVariable<float>(Z, agent_pos[2]);
    FLAMEGPU->setVariable<float>(VELX, agent_vel[0]);
    FLAMEGPU->setVariable<float>(VELY, agent_vel[1]);
    FLAMEGPU->setVariable<float>(VELZ, agent_vel[2]);

    return ALIVE;
}


void define_environment_submodule(ModelDescription &smm) {
    EnvironmentDescription env = smm.Environment();

    env.newProperty<unsigned short>(RUN_IDX, 0);
    env.newMacroProperty<float, TOTAL_AGENTS_ESTIMATION, SOLUTION_LENGTH>(INTERMEDIATE_TARGET_X);
    env.newMacroProperty<float, TOTAL_AGENTS_ESTIMATION, SOLUTION_LENGTH>(INTERMEDIATE_TARGET_Y);
    env.newMacroProperty<float, TOTAL_AGENTS_ESTIMATION, SOLUTION_LENGTH>(INTERMEDIATE_TARGET_Z);
    env.newMacroProperty<unsigned int, TOTAL_AGENTS_ESTIMATION, SOLUTION_LENGTH>(STAY);
    env.newMacroProperty<short, FLOORS, ENV_DIM_Z, ENV_DIM_X>(COORD2INDEX);

}

void define_message_submodule(ModelDescription &smm) {
    // Location pedestrian message
    MessageSpatial3D::Description pedestrian_message = smm.newMessage<MessageSpatial3D>("location_submodule");
    pedestrian_message.newVariable<id_t>(ID);
    pedestrian_message.newVariable<short>(CONTACTS_ID);
    pedestrian_message.setRadius(3.0f);
    pedestrian_message.setMin(0, 0, 0);
    pedestrian_message.setMax(ENV_DIM_X, ENV_DIM_Y, ENV_DIM_Z);
}



void define_agent_submodule(ModelDescription &smm) {
    AgentDescription pedestrian_sm = smm.newAgent("pedestrian_submodule");

    // Variables
    pedestrian_sm.newVariable<float>(X);
    pedestrian_sm.newVariable<float>(Y);
    pedestrian_sm.newVariable<float>(Z);
    pedestrian_sm.newVariable<float>(ANIMATE);
    pedestrian_sm.newVariable<int>(ANIMATE_DIR, 1);
    pedestrian_sm.newVariable<float>(VELX);
    pedestrian_sm.newVariable<float>(VELY);
    pedestrian_sm.newVariable<float>(VELZ);
    pedestrian_sm.newVariable<float>(STEER_X);
    pedestrian_sm.newVariable<float>(STEER_Y);
    pedestrian_sm.newVariable<float>(STEER_Z);
    pedestrian_sm.newVariable<int>(CAN_MOVE);
    pedestrian_sm.newVariable<short>(CONTACTS_ID, -1);
    pedestrian_sm.newVariable<unsigned short>(TARGET_INDEX);
    pedestrian_sm.newVariable<unsigned short>(NEXT_INDEX);

    AgentFunctionDescription output_location = smm.Agent("pedestrian_submodule").newFunction("outputPedestrianLocationSub", outputPedestrianLocationSub);
    output_location.setMessageOutput("location_submodule");

    AgentFunctionDescription move = smm.Agent("pedestrian_submodule").newFunction("move_agent_function", move_agent_function);
    move.setMessageOutputOptional(true);

    AgentFunctionDescription avoid = smm.Agent("pedestrian_submodule").newFunction("avoid_pedestrians", avoid_pedestrians);
    avoid.setMessageInput("location_submodule");
    avoid.setMessageOutputOptional(true);
}

void define_layer_submodule(ModelDescription &smm) {
    {   
        LayerDescription layer = smm.newLayer();
        layer.addAgentFunction(outputPedestrianLocationSub);
    }

    {
        LayerDescription layer1 = smm.newLayer();
        layer1.addAgentFunction(avoid_pedestrians);
    }

    {
        LayerDescription layer = smm.newLayer();
        layer.addAgentFunction(move_agent_function);
    }
}


SubModelDescription create_smm(ModelDescription &model) {

    ModelDescription sub_model_move("move_agent_submodule");

    define_environment_submodule(sub_model_move);
    define_message_submodule(sub_model_move);
    define_agent_submodule(sub_model_move);
    define_layer_submodule(sub_model_move);

    SubModelDescription smm = model.newSubModel("move", sub_model_move);
    smm.setMaxSteps(STEP);

    smm.SubEnvironment().mapProperty(RUN_IDX, RUN_IDX);
    smm.SubEnvironment().mapMacroProperty(INTERMEDIATE_TARGET_X, INTERMEDIATE_TARGET_X);
    smm.SubEnvironment().mapMacroProperty(INTERMEDIATE_TARGET_Y, INTERMEDIATE_TARGET_Y);
    smm.SubEnvironment().mapMacroProperty(INTERMEDIATE_TARGET_Z, INTERMEDIATE_TARGET_Z);
    smm.SubEnvironment().mapMacroProperty(STAY, STAY);
    smm.SubEnvironment().mapMacroProperty(COORD2INDEX, COORD2INDEX);

    SubAgentDescription asmm = smm.bindAgent("pedestrian_submodule", "pedestrian", true, true);

    return smm;
}


#endif