<script lang="ts">
	import favicon from '$lib/assets/favicon.svg';
	import ModeToggle from '$lib/components/ModeToggle.svelte';
	import { ModeWatcher } from 'mode-watcher';
	import './layout.css';

	let { children } = $props();
</script>

<svelte:head>
	<link rel="icon" href={favicon} />
	<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css" />
</svelte:head>

<ModeWatcher />
<div class="absolute top-4 right-4">
	<ModeToggle />
</div>
<main class="h-full p-6">
	{@render children()}
</main>

<style>
	:global(html, body) {
		height: 100%;
	}

	:global(button) {
		cursor: pointer;
	}

	:global(input[type='number']) {
		appearance: none;
		-moz-appearance: textfield;
	}

	:global(
		input[type='number']::-webkit-outer-spin-button,
		input[type='number']::-webkit-inner-spin-button
	) {
		-webkit-appearance: none;
		margin: 0;
	}
</style>
