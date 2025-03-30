from typing import List

# Dummy model loader and predictor for now
# Replace these with your actual coldStart, DLRM, collaborative model logic

async def recommend_books(user_id: int) -> List[int]:
    # Placeholder: decide which model to use based on user_id or user data
    if user_id < 1000:
        return await cold_start_recommendation(user_id)
    elif user_id % 2 == 0:
        return await dlrm_recommendation(user_id)
    else:
        return await collaborative_recommendation(user_id)


async def cold_start_recommendation(user_id: int) -> List[int]:
    # Dummy implementation
    return [1, 2, 3, 4, 5]


async def dlrm_recommendation(user_id: int) -> List[int]:
    # Dummy implementation
    return [6, 7, 8, 9, 10]


async def collaborative_recommendation(user_id: int) -> List[int]:
    # Dummy implementation
    return [11, 12, 13, 14, 15]