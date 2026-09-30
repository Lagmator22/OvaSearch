'use strict';

const TAX_RATE = 0.18;

/**
 * Add an item to the shopping cart, merging quantities for the same sku.
 */
function addItem(cart, sku, qty) {
  const existing = cart.items.find((it) => it.sku === sku);
  if (existing) {
    existing.qty += qty;
  } else {
    cart.items.push({ sku, qty });
  }
  return cart;
}

/**
 * Remove an item from the cart completely.
 */
function removeItem(cart, sku) {
  cart.items = cart.items.filter((it) => it.sku !== sku);
  return cart;
}

/**
 * Compute the cart total including tax.
 */
function computeTotal(cart, prices) {
  let subtotal = 0;
  for (const it of cart.items) {
    subtotal += prices[it.sku] * it.qty;
  }
  return subtotal * (1 + TAX_RATE);
}

module.exports = { addItem, removeItem, computeTotal };
