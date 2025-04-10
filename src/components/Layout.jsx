import React from "react";
import Header from "./Header";
import { Outlet } from "react-router-dom";

function Layout() {
  return (
    <>
      <div className="nav">
        <Header />
      </div>
      <Outlet />
    </>
  );
}

export default Layout;
