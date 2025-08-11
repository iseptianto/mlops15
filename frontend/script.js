document.addEventListener('DOMContentLoaded', () => {
    const API_BASE_URL = '/api';

    // Helper function to fetch data and display results
    async function fetchData(endpoint, resultElement) {
        resultElement.textContent = 'Loading...';
        try {
            const response = await fetch(`${API_BASE_URL}${endpoint}`);
            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Terjadi kesalahan pada server.');
            }

            resultElement.textContent = JSON.stringify(data, null, 2);
        } catch (error) {
            resultElement.textContent = `Error: ${error.message}`;
        }
    }

    // 1. Recommend for User
    const btnRecommendUser = document.getElementById('btn-recommend-user');
    const inputRecUserId = document.getElementById('rec-user-id');
    const resultRecUser = document.getElementById('result-recommend-user');

    btnRecommendUser.addEventListener('click', () => {
        const userId = inputRecUserId.value;
        if (!userId) {
            alert('User ID wajib diisi.');
            return;
        }
        fetchData(`/recommend/user?user_id=${userId}`, resultRecUser);
    });

    // 2. Hybrid Recommendation
    const btnHybrid = document.getElementById('btn-hybrid');
    const inputHybridUserId = document.getElementById('hybrid-user-id');
    const resultHybrid = document.getElementById('result-hybrid');

    btnHybrid.addEventListener('click', () => {
        const userId = inputHybridUserId.value;
        if (!userId) {
            alert('User ID wajib diisi.');
            return;
        }
        fetchData(`/recommend/hybrid?user_id=${userId}`, resultHybrid);
    });

    // 3. Similar Places
    const btnSimilar = document.getElementById('btn-similar');
    const inputSimilarPlaceName = document.getElementById('similar-place-name');
    const resultSimilar = document.getElementById('result-similar');

    btnSimilar.addEventListener('click', () => {
        const placeName = inputSimilarPlaceName.value;
        if (!placeName) {
            alert('Nama tempat wajib diisi.');
            return;
        }
        fetchData(`/places/similar?place_name=${encodeURIComponent(placeName)}`, resultSimilar);
    });

    // 4. Nearby Places
    const btnNearby = document.getElementById('btn-nearby');
    const inputLat = document.getElementById('nearby-lat');
    const inputLon = document.getElementById('nearby-lon');
    const inputRadius = document.getElementById('nearby-radius');
    const resultNearby = document.getElementById('result-nearby');

    btnNearby.addEventListener('click', () => {
        const lat = inputLat.value;
        const lon = inputLon.value;
        const radius = inputRadius.value;

        if (!lat || !lon) {
            alert('Latitude dan Longitude wajib diisi.');
            return;
        }
        fetchData(`/places/nearby?lat=${lat}&lon=${lon}&radius=${radius}`, resultNearby);
    });

    // 5. User Profile
    const btnProfile = document.getElementById('btn-profile');
    const inputProfileUserId = document.getElementById('profile-user-id');
    const resultProfile = document.getElementById('result-profile');

    btnProfile.addEventListener('click', () => {
        const userId = inputProfileUserId.value;
        if (!userId) {
            alert('User ID wajib diisi.');
            return;
        }
        fetchData(`/user/profile?user_id=${userId}`, resultProfile);
    });
});

