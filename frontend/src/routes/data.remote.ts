import { query } from '$app/server';

export const getCheckpoints = query(async () => {
	const response = await fetch('http://localhost:8080/api/checkpoints');
	return response.json();
});
