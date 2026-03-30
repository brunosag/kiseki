<script lang="ts">
	import './layout.css';
	import favicon from '$lib/assets/favicon.svg';
	import { ModeWatcher } from 'mode-watcher';

	let { children } = $props();
</script>

<svelte:head>
	<link rel="icon" href={favicon} />
</svelte:head>

<ModeWatcher />
<main class="h-full bg-zinc-900 p-6">
	{@render children()}
</main>

<style>
	:global(html, body) {
		height: 100%;
	}

	:global(button) {
		cursor: pointer;
	}
</style>
